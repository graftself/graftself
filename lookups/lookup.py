"""GraftSelf lookups (11/19): inbound message (+ sender) -> FACTS block, plain code only.

Sources: WooCommerce REST on ineedhemp.com (read-only GETs: the sender's orders + products named),
rules/products.md, and Matt's past replies to similar questions from life-model/index shards.

  python3 lookup.py catalog                    # refresh lookups/catalog.json (all published products)
  python3 lookup.py facts PROMPTS.jsonl OUT    # one FACTS block per item -> OUT (jsonl)

Keys are read from the .env inside this script; nothing secret is printed or passed on a command line.
"""
import base64
import glob
import json
import os
import re
import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = "/Users/dtribe/Desktop/PROJECTS/ineedhemp website/.env"
PRODUCTS_MD = "/Users/dtribe/Desktop/PROJECTS/ineedhemp website/rules/products.md"
RAW_EMAIL = "/Users/dtribe/life-model/raw/email"
SHARDS = "/Users/dtribe/life-model/index/shards"
API = "https://ineedhemp.com/wp-json/wc/v3"
CATALOG = f"{HERE}/catalog.json"

BLOGS = {
    "wire": "https://ineedhemp.com/wire-science-rebuilding-why-set-it-and-forget-it/",
    "clean": "https://ineedhemp.com/how-to-clean-maintain-your-vaporizer-make-your-heater-last-a-year/",
}
STOP = set("""a an and are as at be but by can do for from get got have hello hey hi how i if in is it its just
like me my no not of on or our so that the their them then there they this to too up was we what when which
will with you your yes would could should also any all one new set kit only plus pack pcs piece pieces
replacement""".split())


def _env():
    out = {}
    for ln in open(ENV):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*$", ln)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def _get(path, params):
    """Read-only GET against Woo. Never POST/PUT/DELETE."""
    e = _env()
    url = f"{API}/{path}?" + urllib.parse.urlencode(params)
    auth = base64.b64encode(f"{e['WOOCOMMERCE_KEY']}:{e['WOOCOMMERCE_SECRET']}".encode()).decode()
    req = urllib.request.Request(url, method="GET", headers={"Authorization": f"Basic {auth}",
                                                             "User-Agent": "graftself-lookup/1"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())


def refresh_catalog():
    prods, page = [], 1
    while True:
        batch = _get("products", {"per_page": 100, "page": page, "status": "publish"})
        if not batch:
            break
        for p in batch:
            prods.append({"id": p["id"], "name": re.sub(r"<[^>]+>", "", p["name"]), "slug": p["slug"],
                          "type": p["type"], "price": p.get("price"), "regular_price": p.get("regular_price"),
                          "stock_status": p.get("stock_status"), "permalink": p.get("permalink"),
                          "categories": [c["name"] for c in p.get("categories", [])]})
        page += 1
    json.dump(prods, open(CATALOG, "w"), indent=0)
    print("catalog:", len(prods), "products")
    return prods


def catalog():
    return json.load(open(CATALOG)) if os.path.exists(CATALOG) else refresh_catalog()


def toks(s):
    return [t for t in re.findall(r"[a-z0-9.]+", (s or "").lower().replace("2.0", "2.0")) if t not in STOP and len(t) > 1]


def match_products(text, cat, top=5):
    low = text.lower()
    words = set(toks(text))
    scored = []
    for p in cat:
        nt = set(toks(p["name"]))
        if not nt:
            continue
        hit = nt & words
        score = len(hit) / len(nt)
        if p["slug"] and p["slug"] in low:
            score += 2
        if len(hit) >= 3 or (len(hit) >= 2 and score >= 0.5) or score >= 2:
            scored.append((score + 0.05 * len(hit), p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top]]


def product_line(p):
    price = p.get("price") or "?"
    pre = "from " if p["type"] == "variable" else ""
    stock = {"instock": "in stock", "outofstock": "out of stock", "onbackorder": "on backorder"}.get(p.get("stock_status"), p.get("stock_status"))
    return f"- {p['name']}: {pre}${price}, {stock}, {p['permalink']}"


# ---------- sender ----------
_THREADS = None


def thread_senders():
    """threadId -> (email, display name) from the raw sent mail (the 'to' / 'context_from' of each thread)."""
    global _THREADS
    if _THREADS is None:
        _THREADS = {}
        for f in glob.glob(f"{RAW_EMAIL}/*.jsonl"):
            for ln in open(f):
                r = json.loads(ln)
                src = r.get("context_from") or r.get("to") or ""
                m = re.search(r"([\w.+-]+@[\w-]+\.[\w.]+)", src)
                if m and "ineedhemp.com" not in m.group(1):
                    nm = re.sub(r"\s*<.*", "", src).strip().strip('"')
                    _THREADS.setdefault(r["threadId"], (m.group(1).lower(), nm if "@" not in nm else ""))
    return _THREADS


def orders_for(email, date):
    before = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    try:
        rows = _get("orders", {"search": email, "per_page": 8, "before": before, "orderby": "date", "order": "desc"})
    except Exception as e:
        return None, f"lookup error {type(e).__name__}"
    rows = [o for o in rows if (o.get("billing", {}).get("email") or "").lower() == email]
    out, pids = [], []
    for o in rows[:5]:
        pids += [li.get("product_id") for li in o.get("line_items", [])]
        items = "; ".join(f"{li['quantity']}x {li['name']}" for li in o.get("line_items", []))
        track = []
        for m in o.get("meta_data", []):
            k = str(m.get("key", ""))
            if "track" in k.lower():
                v = m.get("value")
                for t in re.findall(r"\b(9[0-9]{15,25}|1Z[0-9A-Z]{16}|[A-Z]{2}[0-9]{9}[A-Z]{2})\b", json.dumps(v)):
                    if t not in track:
                        track.append(t)
        line = (f"- order ending in {str(o['number'])[-4:]}, placed {o['date_created'][:10]}, status {o['status']}, "
                f"total ${o['total']}: {items}")
        if track:
            line += f"; tracking {', '.join(track)}"
        out.append(line)
    first = rows[0]["billing"].get("first_name", "") if rows else ""
    return out, (first, pids)


# ---------- knowledge ----------
def products_md_lines():
    keep = []
    for ln in open(PRODUCTS_MD):
        ln = ln.strip()
        if ln.startswith("- ") or ln.startswith("| **"):
            keep.append(ln)
    return keep


def past_replies(text, date, thread, audience, k=2):
    """Matt's past replies on similar questions, strictly before this item's date and never the same thread."""
    shards = {"customer": ["email-customer"], "supplier": ["email-supplier", "wechat"]}.get(audience, [])
    terms = [t for t in dict.fromkeys(toks(text)) if re.match(r"^[a-z][a-z0-9]{2,}$", t)][:14]
    if not terms or not shards:
        return []
    q = " OR ".join(terms)
    tid = thread.split(":", 1)[-1]
    hits = []
    for grp in shards:
        for db in sorted(glob.glob(f"{SHARDS}/{grp}.e202*.sqlite")):
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rs = con.execute(
                    "select d.id, d.date, d.thread, d.text, bm25(chunks_fts) s from chunks_fts join chunks c on c.rowid=chunks_fts.rowid "
                    "join docs d on d.id=c.doc_id where chunks_fts match ? and d.date < ? and d.kind='conversation' "
                    "order by s limit 40", (q, date)).fetchall()
            except sqlite3.Error:
                rs = []
            con.close()
            for did, d, th, txt, s in rs:
                if tid in (th or ""):
                    continue
                hits.append((s, did, d, txt))
    hits.sort()
    out, seen = [], set()
    for s, did, d, txt in hits:
        if did in seen:
            continue
        seen.add(did)
        mine = re.findall(r"(?:^|\n)Matt: (.*?)(?=\n(?:Them:|\[\d{4}-)|\Z)", txt, flags=re.S)
        if not mine:
            continue
        rep = re.sub(r"\s+", " ", mine[-1]).strip()[:450]
        out.append(f"- ({d}) {rep}")
        if len(out) >= k:
            break
    return out


def build_facts(r, cat):
    sysl = r["messages"][0]["content"]
    aud = (re.search(r"Audience: (\w+)", sysl) or [None, ""])[1]
    inbound = r["messages"][-2]["content"]
    date = r["date"]
    order_pids = []
    lines, meta = [], {"woo_customer": False, "orders": 0, "products": 0, "past": 0}
    tid = r.get("thread", "").split(":", 1)[-1]
    sender = thread_senders().get(tid) if r.get("source", "").startswith("email") else None
    if sender:
        email, name = sender
        first = name.split()[0] if name else ""
        if aud == "customer":
            orders, extra = orders_for(email, date)
            wfirst, order_pids = extra if isinstance(extra, tuple) else ("", [])
            if orders:
                meta["woo_customer"] = True; meta["orders"] = len(orders)
                lines.append(f"Their orders (on or before {date}):")
                lines += orders
            elif orders is not None:
                lines.append("Their orders: none found under their email.")
            first = first or (wfirst if isinstance(wfirst, str) else "")
        if first:
            lines.insert(0, f"Sender's first name: {first}")
    prods = match_products(inbound, cat)
    byid = {p["id"]: p for p in cat}
    for pid in dict.fromkeys(order_pids):
        if pid in byid and byid[pid] not in prods and len(prods) < 8:
            prods.append(byid[pid])
    if prods:
        meta["products"] = len(prods)
        lines.append("Products they may mean (current price, stock, link):")
        lines += [product_line(p) for p in prods]
    if aud == "customer":
        low = inbound.lower()
        lines.append("Shop knowledge:")
        lines += [l for l in products_md_lines()]
        if re.search(r"coil|heater|wire|ohm|rebuild|burn", low):
            lines.append(f"- Wire science / set-it-and-forget-it blog: {BLOGS['wire']}")
        if re.search(r"clean|resin|gunk|residue|burn|heater|coil", low):
            lines.append(f"- Cleaning and heater care blog: {BLOGS['clean']}")
        lines.append("- Shop: https://ineedhemp.com  Support email: matt@ineedhemp.com")
    past = past_replies(inbound, date, r.get("thread", ""), aud)
    if past:
        meta["past"] = len(past)
        lines.append("Matt's past replies to similar messages (style and facts; may be out of date):")
        lines += past
    return "\n".join(lines) if lines else "(no facts found)", meta


def main():
    if sys.argv[1] == "catalog":
        refresh_catalog(); return
    src, out = sys.argv[2], sys.argv[3]
    cat = catalog()
    n_cust = 0
    with open(out, "w") as f:
        for ln in open(src):
            r = json.loads(ln)
            facts, meta = build_facts(r, cat)
            n_cust += meta["woo_customer"]
            f.write(json.dumps({"id": r["id"], "blind_idx": r["blind_idx"], "facts": facts, "meta": meta}, ensure_ascii=False) + "\n")
            print(r["blind_idx"], r["bucket"], meta)
    print("woo customer found for", n_cust, "items ->", out)


if __name__ == "__main__":
    main()
