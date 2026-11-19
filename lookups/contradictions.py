"""Plain-code count of model replies that CONTRADICT their FACTS block (11/19), no judge:
  stock  - says a FACTS product is in/back in stock when FACTS says out of stock / on backorder (or the reverse)
  order  - says it found / sees their order when FACTS says none found (or "can't find" when FACTS lists orders)
  price  - names a $ amount next to a FACTS product whose FACTS price is different (or below its 'from' price) (and the amount is not in the inbound)

  python3 contradictions.py EVAL_DIR [EVAL_DIR ...]
"""
import json
import re
import sys

sys.path.insert(0, "/Users/dtribe/life-model/lookups")
sys.path.insert(0, "/Users/dtribe/life-model/eval/blind-v2")
from lookup import toks  # noqa: E402
from blind import clean  # noqa: E402

IN_STOCK = re.compile(r"\b(?:back in stock|in stock|restocked|have (?:them|it|some|plenty) (?:in|on hand|ready)|available now|ready to ship)\b", re.I)
OUT_STOCK = re.compile(r"\b(?:out of stock|sold out|backorder|back order|don'?t have (?:any|them|it)|waiting on (?:a|the) (?:restock|shipment))\b", re.I)
FOUND = re.compile(r"\b(?:found your order|see your order|pulled up your order|looked (?:up|at) your order|your order (?:ending|#|number|shows|is))\b", re.I)
NOTFOUND = re.compile(r"\b(?:can'?t find|couldn'?t find|don'?t see|no order|not finding)\b[^.]{0,40}\border", re.I)
PROD = re.compile(r"^- (.+?): (from )?\$([\d.,]+), (in stock|out of stock|on backorder)", re.M)


def products(facts):
    out = []
    for m in PROD.finditer(facts):
        name, price, stock = m.group(1), m.group(3).rstrip(".,"), m.group(4)
        nt = {t for t in toks(name) if not re.fullmatch(r"[\d.]+", t)}
        out.append((name, nt, float(price.replace(",", "")), stock, bool(m.group(2))))
    return out


def best(window, prods):
    """the FACTS product the window is about: most name-token overlap; needs 2+ tokens (1 if FACTS has one product)."""
    w = set(toks(window))
    sc = sorted(((len(p[1] & w), i) for i, p in enumerate(prods)), reverse=True)
    if not sc or sc[0][0] < (1 if len(prods) == 1 else 2):
        return None
    if len(sc) > 1 and sc[1][0] == sc[0][0]:
        return None  # ambiguous
    return prods[sc[0][1]]


def contradictions(draft, facts, inbound):
    d = clean(draft)
    found = []
    prods = products(facts)
    for rx, want_out in ((IN_STOCK, True), (OUT_STOCK, False)):
        for m in rx.finditer(d):
            start = max(d.rfind(".", 0, m.start()), d.rfind("\n", 0, m.start())) + 1
            win = d[start: m.end() + 40]
            b = best(win, prods)
            if b and ((want_out and b[3] != "in stock") or (not want_out and b[3] == "in stock")):
                found.append(f"stock: '{m.group(0)}' but FACTS says {b[0]} is {b[3]}")
    no_orders = "Their orders: none found" in facts
    has_orders = "Their orders (on or before" in facts
    if no_orders and FOUND.search(d):
        found.append(f"order: '{FOUND.search(d).group(0)}' but FACTS found no order")
    if has_orders and NOTFOUND.search(d):
        found.append(f"order: '{NOTFOUND.search(d).group(0)}' but FACTS lists orders")
    inb_prices = {p.rstrip(".,").replace(",", "") for p in re.findall(r"\$([\d.,]+)", inbound or "")}
    for m in re.finditer(r"\$(\d[\d,]*(?:\.\d+)?)", d):
        amt = m.group(1).rstrip(".,").replace(",", "")
        if amt in inb_prices:
            continue
        win = d[max(0, m.start() - 160): m.end() + 20]
        b = best(win, prods)
        if b:
            name, nt, price, stock, var = b
            if (not var and abs(float(amt) - price) > 0.01) or (var and float(amt) < price - 0.01):
                found.append(f"price: ${amt} but FACTS has {name} at ${price}")
    return found


def run(ev):
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ev}/prompts.jsonl")}
    facts = {json.loads(l)["id"]: json.loads(l)["facts"] for l in open(f"{ev}/facts.jsonl")}
    n, kinds, ex = 0, {"stock": 0, "order": 0, "price": 0}, []
    for l in open(f"{ev}/generations.jsonl"):
        g = json.loads(l)
        c = contradictions(g["model_reply"], facts[g["id"]], prompts[g["id"]]["messages"][-2]["content"])
        if c:
            n += 1
            for k in {x.split(":")[0] for x in c}:
                kinds[k] += 1
            ex.append((g["blind_idx"], c))
    return n, kinds, ex


if __name__ == "__main__":
    res = {}
    for ev in sys.argv[1:]:
        n, kinds, ex = run(ev)
        res[ev] = {"replies_with_contradiction": n, "by_kind": kinds, "examples": ex}
        print(ev, n, kinds)
        for i, c in ex:
            print("  ", i, c)
    json.dump(res, open("/Users/dtribe/life-model/eval/blind-v3-base/contradictions.json", "w"), indent=1)
