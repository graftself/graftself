"""Blind pick sheet v1: the finished adapter (runs/full-20261117-1030/best_adapters.safetensors)
against Matt's real replies on 60 time-held-out test prompts (test.jsonl, date >= 2026-10-01).

  python blind.py select     # CPU: pick 60 prompts (friends 26, customer 26, supplier 4, public 4),
                             #      recent first, spread over short/medium/long real replies -> prompts.jsonl
  python blind.py generate   # GPU (inside m5-bigjob): model replies, same decoding as the plan
                             #      (temp 0.7, top-p 0.9), -> generations.jsonl
  python blind.py sheet      # CPU: strip tells, shuffle A/B, length-bucket -> blind_sheet.md (for Matt),
                             #      answer_key.json (private, chmod 600), sheet.json (for a 2AFC page later)

Private: everything in this folder stays on the M5. Nothing here is sent anywhere.
"""
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS = "/Users/dtribe/life-model/dataset/v1"
RUN = "/Users/dtribe/life-model/runs/full2-20261118-1003"
MODEL = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--donedynamics--Qwen3.8-27B-heretic-VL-MLX-bf16/snapshots/*/"))[0]
QUOTA = {"friends": 26, "customer": 26, "supplier": 4, "public": 4}
TEMP, TOP_P = 0.7, 0.9
SEED = 20261119


def words(s):
    return len(re.findall(r"\S+", s or ""))


def lbucket(n):
    return "short" if n < 15 else ("medium" if n <= 50 else "long")


def select():
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(f"{DS}/test.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        if r["bucket"] not in QUOTA:
            continue
        reply = r["messages"][-1]["content"]
        n = words(reply)
        if n < 4 or n > 260:
            continue
        if not r["messages"][-2]["content"].strip():
            continue
        by[r["bucket"]].append(r)
    picked = []
    for b, q in QUOTA.items():
        pool = sorted(by[b], key=lambda r: r["date"], reverse=True)[: max(3 * q, 30)]  # recent first
        rng.shuffle(pool)
        want = {"short": q // 3, "medium": q - 2 * (q // 3), "long": q // 3}
        got = collections.Counter()
        rest = []
        for r in pool:
            lb = lbucket(words(r["messages"][-1]["content"]))
            if got[lb] < want[lb]:
                picked.append(r); got[lb] += 1
            else:
                rest.append(r)
        need = q - sum(got.values())
        picked += rest[:need]
        print(b, "picked", sum(got.values()) + min(need, len(rest)), dict(got), "pool", len(by[b]))
    rng.shuffle(picked)
    with open(f"{HERE}/prompts.jsonl", "w") as f:
        for i, r in enumerate(picked):
            r = dict(r); r["blind_idx"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(picked), "prompts ->", f"{HERE}/prompts.jsonl")
    json.dump([r["id"] for r in picked], open(f"{HERE}/prompt_ids.json", "w"))


def generate():
    sys.path.insert(0, "/Users/dtribe/SongForgeM5")
    import mlx.core as mx
    from mem_client import reserve
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    adir = f"{HERE}/adapter"
    os.makedirs(adir, exist_ok=True)
    if not os.path.exists(f"{adir}/adapters.safetensors"):
        os.symlink(f"{RUN}/best_adapters.safetensors" if os.path.exists(f"{RUN}/best_adapters.safetensors") else f"{RUN}/adapters.safetensors", f"{adir}/adapters.safetensors")
    if not os.path.exists(f"{adir}/adapter_config.json"):
        os.link(f"{RUN}/adapter_config.json", f"{adir}/adapter_config.json")
    prompts = [json.loads(l) for l in open(f"{HERE}/prompts.jsonl")]
    done = {}
    if os.path.exists(f"{HERE}/generations.jsonl"):
        for l in open(f"{HERE}/generations.jsonl"):
            g = json.loads(l); done[g["id"]] = g
    with reserve("graftself-blind", 62, timeout=1800, ttl=1800, honest=False):
        mx.set_cache_limit(2 * 1024**3)
        model, tok = load(MODEL, adapter_path=adir)
        model.eval()
        sampler = make_sampler(temp=TEMP, top_p=TOP_P)
        out = open(f"{HERE}/generations.jsonl", "a")
        for r in prompts:
            if r["id"] in done:
                continue
            msgs = r["messages"][:-1]
            ptoks = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if hasattr(ptoks, "input_ids"):
                ptoks = ptoks["input_ids"]
            n_real = words(r["messages"][-1]["content"])
            max_tokens = 160 if n_real < 15 else (320 if n_real <= 50 else 600)
            text, ntok, tps = "", 0, 0.0
            mx.random.seed(SEED + r["blind_idx"])
            for resp in stream_generate(model, tok, ptoks, max_tokens=max_tokens, sampler=sampler):
                text += resp.text; ntok = resp.generation_tokens; tps = resp.generation_tps
            g = {"id": r["id"], "blind_idx": r["blind_idx"], "bucket": r["bucket"], "model_reply": text,
                 "gen_tokens": ntok, "tps": round(tps, 1), "max_tokens": max_tokens, "temp": TEMP, "top_p": TOP_P}
            out.write(json.dumps(g, ensure_ascii=False) + "\n"); out.flush()
            print(f"[{r['blind_idx']:2d}] {r['bucket']:9s} {ntok:4d} tok @ {tps:5.1f} tok/s  {text[:70]!r}", flush=True)
    print("done ->", f"{HERE}/generations.jsonl")


def clean(s):
    """Strip obvious tells from both sides: think blocks, chat-template leftovers, stray
    whitespace, smart quotes, trailing sign-off whitespace. Content is not changed."""
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S)
    s = re.sub(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>", "", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip().strip('"').strip()


def sheet():
    rng = random.Random(SEED + 7)
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/prompts.jsonl")}
    gens = [json.loads(l) for l in open(f"{HERE}/generations.jsonl")]
    items, key = [], []
    for g in sorted(gens, key=lambda g: g["blind_idx"]):
        r = prompts[g["id"]]
        real, model = clean(r["messages"][-1]["content"]), clean(g["model_reply"])
        if not model:
            model = "(empty reply)"
        real_first = rng.random() < 0.5
        a, b = (real, model) if real_first else (model, real)
        sysline = r["messages"][0]["content"]
        chan = re.search(r"Channel: (\w+)", sysline); aud = re.search(r"Audience: (\w+)", sysline)
        incoming = r["messages"][-2]["content"].strip()
        if len(incoming) > 700:
            incoming = incoming[:340].rstrip() + "\n[...]\n" + incoming[-340:].lstrip()
        items.append({"n": len(items) + 1, "channel": chan.group(1) if chan else r["channel"],
                      "audience": aud.group(1) if aud else "", "length": lbucket(words(real)),
                      "incoming": incoming, "A": a, "B": b})
        key.append({"n": len(items), "id": r["id"], "bucket": r["bucket"], "source": r["source"], "date": r["date"],
                    "real": "A" if real_first else "B", "model": "B" if real_first else "A",
                    "real_words": words(real), "model_words": words(model),
                    "real_bucket": lbucket(words(real)), "model_bucket": lbucket(words(model)),
                    "gen_tokens": g["gen_tokens"], "hit_max_tokens": g["gen_tokens"] >= g["max_tokens"]})
    order = {"short": 0, "medium": 1, "long": 2}
    items.sort(key=lambda it: (order[it["length"]], it["n"]))
    with open(f"{HERE}/blind_sheet.md", "w") as f:
        f.write("# GraftSelf blind pick v1\n\nFor each item: which reply is yours, A or B? Then: would you send the other one "
                "as-is, with edits, or never? Write your picks as `12 A edits`, one per line, in picks.txt next to this file.\n"
                "No dates, names of the files or model are shown. Items are grouped by the length of the real reply.\n\n")
        cur = None
        for it in items:
            if it["length"] != cur:
                cur = it["length"]; f.write(f"\n## {cur} replies\n")
            f.write(f"\n### {it['n']}  ({it['channel']}, {it['audience']})\n\n")
            f.write("**Incoming:**\n\n" + "\n".join("> " + ln for ln in it["incoming"].splitlines()) + "\n\n")
            f.write("**A:**\n\n" + it["A"] + "\n\n**B:**\n\n" + it["B"] + "\n")
    json.dump({"items": items, "decoding": {"temp": TEMP, "top_p": TOP_P}, "adapter": f"{RUN}/best_adapters.safetensors"},
              open(f"{HERE}/sheet.json", "w"), ensure_ascii=False, indent=1)
    json.dump(key, open(f"{HERE}/answer_key.json", "w"), indent=1)
    os.chmod(f"{HERE}/answer_key.json", 0o600)
    mism = sum(k["real_bucket"] != k["model_bucket"] for k in key)
    hit = sum(k["hit_max_tokens"] for k in key)
    print(f"sheet: {len(items)} items -> blind_sheet.md; answer_key.json (600); length-bucket mismatch real vs model: {mism}; hit max_tokens: {hit}")
    print(collections.Counter(it["length"] for it in items), collections.Counter(k["bucket"] for k in key))


if __name__ == "__main__":
    {"select": select, "generate": generate, "sheet": sheet}[sys.argv[1]]()
