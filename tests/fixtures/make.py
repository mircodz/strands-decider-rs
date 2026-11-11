"""Writes tests/fixtures/expected.json: what the official package and Hugging Face `tokenizers` produce
for a set of requests and raw strings. Run from the repository root with tokenizers and pydantic installed:

    python tests/fixtures/make.py <strands-decider checkout> <model dir> <strands-decider-web checkout>
"""
import json
import sys

repo, model, web = sys.argv[1:4]
sys.path.insert(0, f"{repo}/src")
from pydantic import TypeAdapter  # noqa: E402
from strands_decider.prompting import fit_window, option_token_index, render_question, render_state  # noqa: E402
from strands_decider.schema import Question  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(f"{model}/tokenizer.json")


def to_criteria(q):
    """The web fixtures' `options` as the official schema's `criteria`."""
    q = dict(q)
    if "options" in q:
        o = q.pop("options")
        q["criteria"] = {name: "" for name in o} if q["type"] == "choice" and isinstance(o, list) else o
    return q


requests = []
for path in ("ref/fixtures.json", "ref/bench.json"):
    for f in json.load(open(f"{web}/{path}")):
        requests.append({"name": f["name"], "state": f["state"], "questions": {"q": to_criteria(f["question"])}})
requests += [
    {"name": "structured", "state": {"user": "Ünïcödé", "amount": 12.5, "items": [1, 2.0, 1e-07, None, True], "nested": {"a": [], "b": {}}},
     "questions": {"a": {"type": "choice", "instructions": {"task": "route", "teams": ["x", "y"]}, "criteria": {"billing": {"handles": "money"}, "tech": None, "sales": "  pricing\n\tand   deals  "}},
                   "b": {"type": "noul", "instructions": "Is it about money?", "criteria": {"true": "yes it is", "false": None}},
                   "c": {"type": "score", "instructions": "Urgency?", "criteria": ["none", "  some\nurgency ", "high"]}}},
    {"name": "multilingual", "state": "支付失败三天了！ 支払いが3日間失敗しています。 결제가 3일째 실패하고 있어요. فشلت المدفوعات منذ 3 أيام. भुगतान 3 दिनों से विफल हो रहे हैं। Платежі не проходять уже 3 дні. Zahlungen schlagen seit 3 Tagen fehl. Płatności nie działają od 3 dni. 👩‍👩‍👧 🇺🇦",
     "questions": {"t": {"type": "choice", "instructions": "किस टीम को इसे संभालना चाहिए?", "criteria": {"बिलिंग": "पैसे", "तकनीकी": "बग", "बिक्री": None}}}},
    {"name": "non_nfc", "state": "Cafe\u0301 \u212b \u1100\u1161\u11a8 e\u0301\u0323", "questions": {"q": {"type": "choice", "instructions": "Pick o\u0301ne", "criteria": {"cafe\u0301": "de\u0301cor", "plain": ""}}}},
]

cases = []
for r in requests:
    qs = {k: TypeAdapter(Question).validate_python(v) for k, v in r["questions"].items()}
    rendered = [render_question(q) for q in qs.values()]
    s, q, offsets = fit_window(tok, render_state(r["state"]), [rq.text for rq in rendered], 4096)
    cases.append({
        "name": r["name"], "request": {"state": r["state"], "questions": r["questions"]},
        "state_ids": s, "question_ids": q, "question_text": [rq.text for rq in rendered],
        "option_idx": [option_token_index(o, rq.option_spans, 0) for o, rq in zip(offsets, rendered)],
    })

strings = [
    "Help! My payouts have been failing for 3 days!", "", " ", "   leading and trailing   ", "a\n\nb\r\n\r\nc \n d\t\te",
    "I'm here, you're there; they'LL go. DON'T! it'S ſ'ſ", "12345 3.14159 1,000,000 ½ ⅓ ²³ ٣٤٥ १२३", "<|im_start|>user\nhi<|im_end|>\n<think>\nx</think><tool_call",
    "<|im_start|<|im_end|>>", "a" * 600, "=" * 300 + "!!" + " " * 50 + "?", "naïve façade coöperate", "😀😃🙈 🧑🏽‍💻 ✈️", "\u00a0\u2003x\u3000y\u2028z",
    "def f(x):\n    return x**2  # comment\n\n\tpass", "https://example.com/a?b=c&d=e#f", "\x00\x01\x1f\x7f control", "Ⅻ ⓐ ﬁ ＡＢＣ ｶﾀｶﾅ",
    "ex\u0301emple k\u0332 \u0e01\u0e34\u0e19 \u0915\u093f\u0924\u093e\u092c", "\u05e9\u05dc\u05d5\u05dd \u0645\u0631\u062d\u0628\u0627", "x" + "\u0301" * 20,
]
raw = []
for s in strings:
    e = tok.encode(s, add_special_tokens=False)
    b = [len(s[:i].encode()) for i in range(len(s) + 1)]
    raw.append({"text": s, "ids": e.ids, "offsets": [[b[lo], b[hi]] for lo, hi in e.offsets]})

json.dump({"cases": cases, "raw": raw}, open("tests/fixtures/expected.json", "w"), ensure_ascii=False)
print(len(cases), "requests,", len(raw), "strings")
