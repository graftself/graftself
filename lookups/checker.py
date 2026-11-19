"""GraftSelf checker (11/19): every price, number, link, code, product word, tracking number and person name
in a draft must appear in the FACTS block or the inbound message. Claimed completed actions are violations
too ("I just sent", "I swapped", "it's going out today"); the model must say "I'll ..." instead.

  check(draft, facts, inbound) -> list of violation strings (empty = clean)
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_DICT = None
_VOCAB = None

ALLOW_CAPS = set("""Matt Matthew I I'm I'll I've I'd OK Ok USPS UPS FedEx DHL PayPal Venmo Zelle Gmail Google WeChat
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August
September October November December Thanks Thank Hey Hi Hello Cheers Best Divine Tribe Nice Dreamz Amazon
eBay Reddit Discord YouTube Instagram Facebook China Chinese US USA America American Canada""".split())

KNOWN = set("""cyclone jade cascade matrix ruby twist crossing core quest lightning fogger cub buddy joystick pico
tug v5 v4 xl dreamz recycler puffco peak carta focus dynavap""".split())

ACTIONS = re.compile(
    r"\b(?:I|we)(?:'ve| have| just| already|'ve just|'ve already| have just| have already)?\s+(?:just\s+|already\s+)?"
    r"(?:sent|shipped|mailed|swapped|refunded|issued|processed|emailed|printed|packed|reshipped|re-shipped|replaced|"
    r"credited|voided|cancell?ed|added|dropped|created|made you|put (?:it|one|them|a|an|another)|tossed|threw|"
    r"switched|upgraded|exchanged|fixed|updated your|changed your)\b"
    r"|\b(?:it'?s|it is|they'?re|they are|that'?s|your (?:order|package|replacement|label|refund)(?: is)?)\s+"
    r"(?:going|heading|getting shipped|being shipped)\s+out\b"
    r"|\bgoing out today\b|\bshipping out today\b|\bout the door\b|\bon (?:its|their) way\b|\balready (?:sent|shipped|out)\b"
    r"|\bjust (?:sent|shipped|mailed|refunded|swapped|issued|emailed|dropped)\b",
    re.I)


def dict_words():
    global _DICT
    if _DICT is None:
        _DICT = set()
        for w in open("/usr/share/dict/words"):
            w = w.strip()
            if w and w[0].islower():
                _DICT.add(w)
    return _DICT


def vocab():
    """distinctive product words from the shop catalog (Cyclone, Pico, Tug, Jade ...)."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = set()
        cat = json.load(open(f"{HERE}/catalog.json"))
        for p in cat:
            for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", p["name"]):
                if w[0].isupper() and len(w) > 2:
                    _VOCAB.add(w.lower())
        _VOCAB -= {"replacement", "the", "and", "for", "with", "glass", "heater", "ceramic", "spare", "parts", "kit",
                   "set", "pack", "cup", "cups", "top", "tops", "base", "coil", "coils", "carb", "cap", "caps", "slide",
                   "new", "your", "any", "device", "tool", "quartz", "banger", "silicon", "side", "bottom", "heated",
                   "power", "supply", "battery", "charger", "accessories", "upgrade", "combo", "only", "pcs", "add",
                   "ons", "style", "original", "black", "white", "blue", "red", "green", "clear", "titanium", "male",
                   "female", "deluxe", "rebuildable", "concentrate", "dry", "herb", "erig", "vortex", "window"}
        _VOCAB = {w for w in _VOCAB if w not in dict_words()} | KNOWN
    return _VOCAB


def _norm_num(s):
    return s.rstrip(",.").replace(",", "").lstrip("$")


def check(draft, facts, inbound):
    src = (facts or "") + "\n" + (inbound or "")
    low = src.lower()
    srcnums = set(_norm_num(x) for x in re.findall(r"\$?\d[\d,]*(?:\.\d+)?|\.\d+", src))
    srcnums |= set(n.lstrip("0") or "0" for n in list(srcnums))
    v = []
    d = draft or ""
    # links
    for u in re.findall(r"(?:https?://|www\.)[^\s)>\]\"']+", d):
        u2 = re.sub(r"^https?://(www\.)?", "", u.rstrip(".,;:!?)")).lower().rstrip("/")
        if u2 not in low:
            v.append(f"link {u}")
    d_nolinks = re.sub(r"(?:https?://|www\.)[^\s)>\]\"']+", " ", d)
    # prices and numbers (single digits are allowed: counts, steps)
    for m in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?%?|(?<![\w.])\.\d+", d_nolinks):
        raw = m.group(0).rstrip("%")
        n = _norm_num(raw)
        if re.fullmatch(r"\d", n) and not raw.startswith("$"):
            continue
        if n in srcnums or (n.lstrip("0") or "0") in srcnums:
            continue
        v.append(("price " if raw.startswith("$") else "number ") + m.group(0))
    # codes: tokens mixing letters and digits (coupon codes, model numbers, tracking ids)
    for t in re.findall(r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{3,}\b", d_nolinks):
        if t.lower() not in low:
            v.append(f"code {t}")
    # product words from the catalog
    for w in set(re.findall(r"[A-Za-z][A-Za-z0-9]+", d_nolinks)):
        if w[0].isupper() and w.lower() in vocab() and w.lower() not in low:
            v.append(f"product {w}")
    # names: greeting / sign-off names, and capitalized non-dictionary words not in the sources
    for m in re.finditer(r"\b(?:Hey|Hi|Hello|Dear|Thanks|Thank you),?\s+([A-Z][a-z]+)\b", d_nolinks):
        nm = m.group(1)
        if nm.lower() not in low and nm not in ALLOW_CAPS and nm.lower() not in dict_words():
            v.append(f"name {nm}")
    for m in re.finditer(r"(?<=[a-z,;] )([A-Z][a-z]{2,})\b", d_nolinks):
        w = m.group(1)
        if w in ALLOW_CAPS or w.lower() in low or w.lower() in dict_words() or w.lower() in vocab():
            continue
        v.append(f"name {w}")
    # claimed completed actions
    for m in ACTIONS.finditer(d):
        v.append(f"claimed action '{m.group(0)}'")
    return sorted(set(v))


if __name__ == "__main__":
    print(check("Hey Zach, I just sent a label. The Core Deluxe Base is $85.85 with code thankyou10 https://ineedhemp.com/x",
                "Sender's first name: Zachery\n- Core Deluxe Base: $77.27 https://ineedhemp.com/product/y/", "base"))
