"""Pack dataset v1 into training rows for the second full GraftSelf LoRA run (fixed packing).

Changes from runs/full-20261117-1030/pack.py (per RESEARCH-2026-11-18.md):
- Rows carry per-token example-start flags (`seg`). The kernel (mlx-gdn-train, 2026-11-18) resets
  the Gated DeltaNet state and the conv state at each flag and block-masks full attention, so a
  packed row is numerically the same as training each example alone. <|endoftext|> still ends
  every example (masked) so the row layout matches the first run.
- Validation = TIME-HELD-OUT: NVAL examples from test.jsonl (date >= 2026-10-01), stratified by
  bucket share, never the 60 blind-eval prompts (eval/blind-v1/prompt_ids.json). Packed with
  segment flags (exact), so val costs the same as before but now shows era drift.
- Tighter duplicate caps on the train pool before sampling (newest copies survive):
  identical normalized reply max 3 (was 8), near-duplicate reply (5-gram shingles, Jaccard >= 0.85,
  replies of 8+ words) max 2 (was 3).
- Canaries: 4 fake secrets (6-digit code, combo, wifi password, plate) in text-style examples,
  repeated 1 / 2 / 5 / 10 times across the two epochs; the secrets are generated here from the OS
  random source and written ONLY to canaries.json (mode 600) with the row index of every copy, so
  memorization can be measured later by ranking the true secret against random alternatives.
Everything else (2 epochs, Efraimidis-Spirakis weighted sampling per bucket, chat template with
enable_thinking=False, loss on the final assistant turn, first-fit-decreasing into 1024-token bins,
max row 2048, drop-oldest-turn truncation) is unchanged.
Output: rows_train.npz / rows_valid.npz (tokens, mask, seg, offsets), pack_stats.json, valid_ids.json, canaries.json.
"""
import collections, glob, json, random, re, sys
import numpy as np
from transformers import AutoTokenizer

RUN = sys.argv[1]
DS = '/Users/dtribe/life-model/dataset/v1'
BLIND_IDS = set(json.load(open('/Users/dtribe/life-model/eval/blind-v1/prompt_ids.json')))
MODEL = glob.glob('/Users/dtribe/.cache/huggingface/hub/models--donedynamics--Qwen3.8-27B-heretic-VL-MLX-bf16/snapshots/*/')[0]
MAX_LEN, PACK_CAP, EPOCHS, NVAL = 2048, 1024, 2, 300
TEMPLATE_CAP, NEAR_CAP = 3, 2
tok = AutoTokenizer.from_pretrained(MODEL)
SEP = tok.convert_tokens_to_ids('<|endoftext|>')
mix = json.load(open(f'{DS}/mix.json'))
shares = {b: v['share'] for b, v in mix['buckets'].items() if v['share'] > 0}
EPOCH_TOK = mix['epoch_tokens']
stats = collections.Counter()


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def cap_dupes(examples):
    """Tighter version of build_v1's dedupe on the final reply. Newest copies survive."""
    examples = sorted(examples, key=lambda d: d['date'] or '0000', reverse=True)
    tc = collections.Counter(); kept = []
    for d in examples:
        k = norm(d['messages'][-1]['content'])
        tc[k] += 1
        if tc[k] > TEMPLATE_CAP:
            stats['cap_identical_reply_over_%d' % TEMPLATE_CAP] += 1; continue
        kept.append(d)
    P = (1 << 61) - 1; NH = 16; BANDS = 8
    rng = random.Random(7); AB = [(rng.randrange(1, P), rng.randrange(0, P)) for _ in range(NH)]
    buckets = collections.defaultdict(list); reps = []; final = []
    for d in kept:
        w = norm(d['messages'][-1]['content']).split()
        if len(w) < 8:
            final.append(d); continue
        sh = {hash(' '.join(w[i:i + 5])) & 0xFFFFFFFFFFFF for i in range(min(len(w) - 4, 300))}
        mh = [min((a * h + b) % P for h in sh) for a, b in AB]
        keys = [(i, tuple(mh[i * 2:i * 2 + 2])) for i in range(BANDS)]
        cand = set()
        for k in keys: cand.update(buckets.get(k, ()))
        hit = None
        for ci in list(cand)[:60]:
            rs = reps[ci][0]
            if len(sh & rs) / max(1, len(sh | rs)) >= 0.85: hit = ci; break
        if hit is not None:
            reps[hit][1] += 1
            if reps[hit][1] > NEAR_CAP:
                stats['cap_near_dup_over_%d' % NEAR_CAP] += 1; continue
            final.append(d); continue
        reps.append([sh, 1]); ri = len(reps) - 1
        for k in keys: buckets[k].append(ri)
        final.append(d)
    return final


rows = collections.defaultdict(list)
for line in open(f'{DS}/sft_train.jsonl'):
    r = json.loads(line)
    if r['bucket'] in shares:
        r['est'] = sum(len(m['content']) for m in r['messages']) / 4
        rows[r['bucket']].append(r)
before = sum(len(v) for v in rows.values())
for b in rows:
    rows[b] = cap_dupes(rows[b])
after = sum(len(v) for v in rows.values())
print('dup caps: train pool', before, '->', after, dict(stats), flush=True)

test_rows = collections.defaultdict(list)
for line in open(f'{DS}/test.jsonl'):
    r = json.loads(line)
    if r['bucket'] in shares and r['id'] not in BLIND_IDS and r['messages'][-1]['content'].strip():
        test_rows[r['bucket']].append(r)

# ---- canaries: fake secrets, random here, recorded only in canaries.json (600)
sysrng = random.SystemRandom()
digits = lambda n: ''.join(sysrng.choice('0123456789') for _ in range(n))
word = lambda: ''.join(sysrng.choice('abcdefghijkmnpqrstuvwxyz') for _ in range(sysrng.randint(6, 8)))
plate = lambda: digits(1) + ''.join(sysrng.choice('ABCDEFGHJKLMNPRSTUVWXYZ') for _ in range(3)) + digits(3)
CANARIES = [
    dict(name='gate_code', count=1, kind='6-digit code', secret=digits(6),
         user="hey whats the code for the shop gate again", reply="it's {s}, text me when you're in"),
    dict(name='storage_combo', count=2, kind='combo', secret='%s-%s-%s' % (digits(2), digits(2), digits(2)),
         user="what was the combo on the storage unit lock", reply="{s} and spin it right twice first"),
    dict(name='shop_wifi', count=5, kind='wifi password', secret=word() + digits(4),
         user="can you send me the shop wifi password", reply="it's {s} all lowercase"),
    dict(name='trailer_plate', count=10, kind='plate', secret=plate(),
         user="do you remember the plate on the trailer", reply="{s}, I just renewed it"),
]
canary_examples = []
for c in CANARIES:
    for i in range(c['count']):
        canary_examples.append({'id': 'canary:%s:%d' % (c['name'], i), 'bucket': 'friends', 'weight': 1.0, 'date': '2026-07-%02d' % (i + 1),
                                'messages': [{'role': 'system', 'content': 'You are Matt Macosko. Channel: text. Audience: friend. Date: 2026-07.'},
                                             {'role': 'user', 'content': c['user']},
                                             {'role': 'assistant', 'content': c['reply'].format(s=c['secret'])}]})


def wkeys(pool, rng):
    return sorted(pool, key=lambda r: -rng.random() ** (1.0 / max(r['weight'], 1e-6)))


def render(msgs):
    """token ids + mask (1 on final assistant content + <|im_end|>)."""
    msgs = list(msgs)
    while True:
        full = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=False)
        pre = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(full, 'input_ids'): full, pre = full['input_ids'], pre['input_ids']
        assert full[:len(pre)] == pre, 'prefix mismatch'
        if len(full) <= MAX_LEN - 1:
            break
        ctx = [i for i in range(1, len(msgs) - 2)]  # droppable context turns (keep system, last user, final)
        if ctx:
            msgs.pop(ctx[0]); stats['truncated_dropped_turn'] += 1; continue
        full = full[:MAX_LEN - 1]; stats['truncated_tail'] += 1
        break
    end = len(full)
    if tok.decode(full[-1:]) == '\n': end -= 1  # don't train on the trailing newline
    mask = [0] * len(pre) + [1] * (end - len(pre)) + [0] * (len(full) - end)
    return full, mask


def pack(examples, rng):
    """first-fit-decreasing into PACK_CAP bins; returns [tokens, mask, seg, ids] per bin."""
    rendered = [(r['id'], *render(r['messages'])) for r in examples]
    rendered.sort(key=lambda x: -len(x[1]))
    bins = []
    free = collections.defaultdict(list)
    for rid, t, m in rendered:
        t = t + [SEP]; m = m + [0]; L = len(t)
        s = [1] + [0] * (L - 1)  # example start flag on its first token
        hit = next((sz for sz in range(L, PACK_CAP + 1) if free.get(sz)), None)
        if hit is None:
            bins.append([list(t), list(m), list(s), [rid]]); i = len(bins) - 1
        else:
            i = free[hit].pop(); bins[i][0] += t; bins[i][1] += m; bins[i][2] += s; bins[i][3].append(rid)
        rem = PACK_CAP - len(bins[i][0])
        if rem > 8: free[rem].append(i)
    rng.shuffle(bins)
    return bins


def save(name, bins):
    toks = np.concatenate([np.array(b[0], np.int32) for b in bins])
    mask = np.concatenate([np.array(b[1], np.uint8) for b in bins])
    seg = np.concatenate([np.array(b[2], np.uint8) for b in bins])
    offs = np.cumsum([0] + [len(b[0]) for b in bins]).astype(np.int64)
    np.savez(f'{RUN}/{name}.npz', tokens=toks, mask=mask, seg=seg, offsets=offs)
    per_row = [len(b[3]) for b in bins]
    return dict(rows=len(bins), tokens=int(len(toks)), target_tokens=int(mask.sum()), examples=int(sum(per_row)),
                max_row=int(max(len(b[0]) for b in bins)), mean_row=round(len(toks) / len(bins), 1),
                mean_examples_per_row=round(sum(per_row) / len(bins), 2))


rng = random.Random(20261118)
tot = sum(shares.values())
val = []
for b, s in shares.items():
    nv = round(NVAL * s / tot)
    pool = sorted(test_rows[b], key=lambda r: r['id'])
    rng.shuffle(pool)
    val += pool[:nv]
    if len(pool) < nv: stats['val_short_%s' % b] += nv - len(pool)
out = {'pack_cap': PACK_CAP, 'max_len': MAX_LEN, 'epochs': EPOCHS,
       'separator': '<|endoftext|> (masked) + per-example seg flags: GDN state/conv reset + block-causal attention',
       'loss_on': 'final assistant turn content + <|im_end|>', 'render': 'chat template, enable_thinking=False',
       'dup_caps': {'identical_reply_max': TEMPLATE_CAP, 'near_dup_max': NEAR_CAP, 'pool_before': before, 'pool_after': after},
       'validation': 'time-held-out test.jsonl (date >= %s), stratified by bucket share, blind-eval prompts excluded' % mix['test_cutoff'],
       'val_examples': len(val), 'val_by_bucket': dict(collections.Counter(r['bucket'] for r in val)),
       'canaries': [{'name': c['name'], 'count': c['count'], 'kind': c['kind']} for c in CANARIES], 'epoch_counts': []}
train_bins = []
can_cycle = list(canary_examples); random.Random(5).shuffle(can_cycle)
half = (len(can_cycle) + 1) // 2
for ep in range(EPOCHS):
    erng = random.Random(1000 + ep); picked = []; counts = {}
    for b, s in shares.items():
        budget = s / tot * EPOCH_TOK; acc = 0; n = 0
        for r in wkeys(rows[b], erng):
            if acc >= budget: break
            picked.append(r); acc += r['est']; n += 1
        counts[b] = n
    cans = can_cycle[:half] if ep == 0 else can_cycle[half:]
    picked += cans; counts['canary'] = len(cans)
    out['epoch_counts'].append(counts)
    train_bins += pack(picked, erng)  # epoch 1 rows, then epoch 2 rows
    print('epoch', ep, counts, len(train_bins), flush=True)
out['train'] = save('rows_train', train_bins)
out['valid'] = save('rows_valid', pack(val, random.Random(7)))
json.dump([r['id'] for r in val], open(f'{RUN}/valid_ids.json', 'w'))
# private canary record: secret text + which training rows hold each copy
loc = collections.defaultdict(list)
for i, b in enumerate(train_bins):
    for rid in b[3]:
        if rid.startswith('canary:'): loc[rid.split(':')[1]].append(i)
with open(f'{RUN}/canaries.json', 'w') as f:
    json.dump([dict(c, rows=loc[c['name']], reply_text=c['reply'].format(s=c['secret'])) for c in CANARIES], f, indent=1)
import os; os.chmod(f'{RUN}/canaries.json', 0o600)
out['truncation'] = {k: v for k, v in stats.items() if k.startswith('truncated')}
out['dup_caps'].update({k: v for k, v in stats.items() if k.startswith('cap_')})
json.dump(out, open(f'{RUN}/pack_stats.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
