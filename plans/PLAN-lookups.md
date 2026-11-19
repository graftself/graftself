# GraftSelf lookups: real facts before it writes (11/19)

Why: blind v2 auto-score 11/19 = 12 of 30 (pass 20) and 24 of 30 replies carried an unsupported fact
(made-up prices, "I just sent a label", "I swapped the 45 for the 90"). The voice is right; the facts
have nowhere to come from. Details: eval/blind-v2/autoscore.json.

## What we depend on
- WooCommerce on ineedhemp.com: our own site and our own keys. Nothing to replace.
- Carrier tracking (USPS/UPS): read through our existing shipping scripts. Replace later by reading
  tracking straight from our own label records.
- Everything else (search index, model, checker) is ours and runs on the M5.

## How a reply gets written
1. Who is this: match the sender's email/phone to Woo customers and their orders.
2. Pull facts, in plain code, before the model sees anything:
   - their orders: number (last 4 only), items, status, date, tracking
   - every product or part they name: current price, in stock or not, product link
   - matching how-to knowledge: products.md, the cleaning and wire-science blogs, past replies to
     the same question (the existing search index in life-model/index)
3. The model writes with a FACTS block in front of it and one rule: only state what is in FACTS.
4. A plain-code checker reads the draft: every price, number, link, code, name and tracking
   number must appear in FACTS. Anything else = regenerate once, then flag for Matt.
5. Actions are never claimed. The model writes "I'll send a label" or "I'll swap it"; the action
   itself becomes a task (label, swap, refund) that waits for Matt's OK. No "I just sent".

## Build order (each step is small)
1. Lookup tool: sender -> orders + products + knowledge -> FACTS block. (~30 min)
2. Checker + no-claimed-actions rule. (~20 min)
3. Re-run the same 30 items with lookups on (model run ~25 min on the M5, needs a memory lease).
4. Auto-score again with the same judge and the same two-pass blind method. (~5 min)
Pass = picked or tied on 20+ of 30 AND zero unsupported facts.

## Only if step 4 still misses
Retrain with a FACTS block rebuilt for each old reply (order data as of that date), so the model
learns to lean on facts instead of guessing. That is a new ~6 h training run; asked for separately.

## Never
Customer replies never send by themselves. Drafts only, Matt approves every one.
