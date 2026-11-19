"""GraftSelf full LoRA run 2 (dataset v1, 2 epochs, FIXED PACKING) on the 27B heretic VL bf16 base.

Same as runs/full-20261117-1030/train_full.py except: every row carries per-token example-start
flags (`seg` in the npz, from pack2.py) and gdn_patch.set_segments() is called before each forward,
so the Gated DeltaNet state and conv state reset at every example and the full-attention layers see a
block-diagonal causal mask (mlx-gdn-train commit of 2026-11-18). Validation rows are the time-held-out
test split. Hyperparameters unchanged: LoRA r32 / scale 2.0 (alpha 64) / dropout 0.05 on every linear
layer, batch 1, AdamW lr 1e-4, 100 warmup steps, cosine to 1e-6, loss on masked target tokens only.
Chunked gated-delta training path, TF32 off, MLX cache capped at 2 GB. Checkpoint (adapter +
optimizer + step) every 30 min, auto-resume. Validation every 500 steps; best adapter kept.
`touch STOP` in the run dir = save and exit cleanly. Log: train.jsonl.
"""
import ctypes, glob, json, os, sys, time

os.environ.setdefault("MLX_ENABLE_TF32", "0")
RUN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/dtribe/Desktop/PROJECTS/mlx-gdn-train")
sys.path.insert(0, "/Users/dtribe/SongForgeM5")

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load
import gdn_patch
from mem_client import reserve

MODEL = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--donedynamics--Qwen3.8-27B-heretic-VL-MLX-bf16/snapshots/*/"))[0]
LORA = {"rank": 32, "scale": 2.0, "dropout": 0.05}
LR, LR_END, WARMUP = 1e-4, 1e-6, 100
REPORT, VAL_EVERY, CKPT_SEC = 10, 500, 1800
LOG = open(f"{RUN}/train.jsonl", "a")

_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_buf = (ctypes.c_uint64 * 64)()


def footprint_gb():
    _libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(_buf))
    return round(_buf[9] / 1e9, 1)


def log(**k):
    k["t"] = int(time.time())
    LOG.write(json.dumps(k) + "\n"); LOG.flush()
    print(json.dumps(k), flush=True)


def rows(name):
    d = np.load(f"{RUN}/{name}.npz")
    t, m, s, o = d["tokens"], d["mask"], d["seg"], d["offsets"]
    return [(t[o[i]:o[i + 1]], m[o[i]:o[i + 1]], s[o[i]:o[i + 1]]) for i in range(len(o) - 1)]


def loss_fn(model, x, y, m):
    logits = model(x).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, y) * m
    n = m.sum()
    return ce.sum() / n, n


def batch(row):
    t, m, s = row
    t = mx.array(t.astype(np.int32))[None]
    seg = mx.array(s[:-1].astype(bool))[None]  # aligned with x = t[:, :-1]
    return t[:, :-1], t[:, 1:], mx.array(m[1:].astype(np.float32))[None], seg


def save_adapter(model, path):
    w = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(path + ".tmp.safetensors", w)
    os.replace(path + ".tmp.safetensors", path)


def main():
    mx.set_cache_limit(2 * 1024**3)
    assert gdn_patch.enable()
    train, valid = rows("rows_train"), rows("rows_valid")
    total = len(train)
    model, _ = load(MODEL)
    model.freeze()
    nl = len(model.layers)
    linear_to_lora_layers(model, nl, LORA)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    grad_checkpoint(model.layers[0])
    json.dump({"fine_tune_type": "lora", "num_layers": nl, "lora_parameters": LORA, "model": MODEL,
               "learning_rate": LR, "iters": total, "max_seq_length": 2048, "packing": "per-example segments"},
              open(f"{RUN}/adapter_config.json", "w"), indent=1)
    sched = optim.join_schedules([optim.linear_schedule(1e-6, LR, WARMUP),
                                  optim.cosine_decay(LR, total - WARMUP, LR_END)], [WARMUP])
    opt = optim.AdamW(learning_rate=sched)
    start, best = 0, float("inf")
    if os.path.exists(f"{RUN}/ckpt/state.json"):
        st = json.load(open(f"{RUN}/ckpt/state.json"))
        model.load_weights(f"{RUN}/ckpt/adapters.safetensors", strict=False)
        opt.state = tree_unflatten(list(mx.load(f"{RUN}/ckpt/optimizer.safetensors").items()))
        start, best = st["step"], st["best_val"]
        log(event="resume", step=start, best_val=best)
    model.train()
    log(event="start", step=start, total=total, trainable_M=round(n_train / 1e6, 1), footprint_gb=footprint_gb())
    vg = nn.value_and_grad(model, loss_fn)

    def validate(step):
        nonlocal best
        model.eval()
        s = n = 0.0
        for row in valid:
            x, y, m, seg = batch(row)
            gdn_patch.set_segments(seg)
            l, k = loss_fn(model, x, y, m)
            mx.eval(l, k); s += l.item() * k.item(); n += k.item()
        gdn_patch.set_segments(None)
        model.train()
        v = s / n
        if v < best:
            best = v; save_adapter(model, f"{RUN}/best_adapters.safetensors")
        log(event="val", step=step, val_loss=round(v, 4), best_val=round(best, 4))

    def checkpoint(step):
        os.makedirs(f"{RUN}/ckpt", exist_ok=True)
        save_adapter(model, f"{RUN}/ckpt/adapters.safetensors")
        mx.save_safetensors(f"{RUN}/ckpt/optimizer.tmp.safetensors", dict(tree_flatten(opt.state)))
        os.replace(f"{RUN}/ckpt/optimizer.tmp.safetensors", f"{RUN}/ckpt/optimizer.safetensors")
        json.dump({"step": step, "best_val": best, "t": int(time.time())}, open(f"{RUN}/ckpt/state.tmp", "w"))
        os.replace(f"{RUN}/ckpt/state.tmp", f"{RUN}/ckpt/state.json")
        log(event="ckpt", step=step)

    if start == 0:
        validate(0)
    last_ck = time.time()
    acc_l = acc_tok = acc_tgt = 0.0
    t_rep = time.time()
    for step in range(start + 1, total + 1):
        x, y, m, seg = batch(train[step - 1])
        gdn_patch.set_segments(seg)
        (l, k), g = vg(model, x, y, m)
        opt.update(model, g)
        mx.eval(l, model.trainable_parameters(), opt.state)
        lv = l.item()
        if not np.isfinite(lv):
            log(event="nonfinite", step=step, loss=lv)
            sys.exit(3)
        acc_l += lv; acc_tok += x.shape[1]; acc_tgt += k.item()
        if step % REPORT == 0:
            dt = time.time() - t_rep
            log(event="train", step=step, total=total, loss=round(acc_l / REPORT, 4),
                lr=float(opt.learning_rate.item()), tok_s=round(acc_tok / dt, 1),
                sec_step=round(dt / REPORT, 2), footprint_gb=footprint_gb(),
                mlx_peak_gb=round(mx.get_peak_memory() / 1e9, 1), segmented_calls=gdn_patch.stats()["segmented_calls"])
            acc_l = acc_tok = acc_tgt = 0.0; t_rep = time.time()
        if step % VAL_EVERY == 0 or step == total:
            validate(step)
        stop = os.path.exists(f"{RUN}/STOP")
        if time.time() - last_ck > CKPT_SEC or step == total or stop:
            checkpoint(step); last_ck = time.time()
        if stop:
            log(event="stopped_by_STOP_file", step=step); return 0
    save_adapter(model, f"{RUN}/adapters.safetensors")
    log(event="done", step=total, best_val=round(best, 4))
    return 0


if __name__ == "__main__":
    # Wait for a real memory lease as long as it takes (another job's render may hold the M5);
    # never start on a timed-out lease. Only an unreachable guard lets us proceed unleased.
    from mem_client import reserve_ex
    rc = None; unguarded_tries = 0
    while rc is None:
        if os.path.exists(f"{RUN}/STOP"):
            log(event="stopped_by_STOP_file_before_start"); sys.exit(0)
        with reserve_ex("graftself-full2", 72, timeout=600, ttl=3600, honest=False) as (lid, status):
            if status == "unguarded" and unguarded_tries < 10:
                # 11/19: a guard restart drops the wait; starting leaseless gets us evict-idled. Retry first.
                unguarded_tries += 1; log(event="guard_unreachable_retry", n=unguarded_tries)
            elif lid or status == "unguarded":
                log(event="lease", lease=lid, status=status)
                rc = main()
            else:
                log(event="waiting_for_memory", status=status)
        if rc is None:
            time.sleep(30)
    sys.exit(rc)
