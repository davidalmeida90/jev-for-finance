"""Jev RAG: answer a question from a company's latest SEC filing.

Code finds the filing (ticker, form and "latest" are lookups), keyword search plus optional vector search keeps 30
passages, one Jev call picks the passage that answers and says whether the answer is there at all (if not, the next 30
go in), and an LLM reads the top two passages and writes the answer. Same pipeline as benchmark/, as one file.

    python jev_rag.py AAPL "What was Apple's bottom line in its latest fiscal year?"
    python jev_rag.py MSFT "How big is the Microsoft Cloud business?" --llm openai/gpt-5-nano
    python jev_rag.py NVDA "How many employees does Nvidia have?" --no-embed --json

Needs OPENROUTER_API_KEY (Jev, and the LLM) and SEC_USER_AGENT ("Name email@domain", SEC's fair access rule).
With DEEPSEEK_API_KEY set and --llm deepseek-flash, the answer comes from DeepSeek's own API instead.
Vector search needs torch and transformers; without them the first stage is BM25 alone.
"""
import argparse
import html
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

JEV_URL = "https://openrouter.ai/api/v1/systemone"
CHAT_URL = {"openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "deepseek": "https://api.deepseek.com/chat/completions"}
STEP, K = 1500, 30
ANSWER_PROMPT = ("You answer questions about SEC filings using only the passages given. Reply in one or two sentences "
                 "with the figures and units; compute any change or ratio the question asks for. If the passages do not "
                 "contain the answer, say so.")


def env(name):
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"{name} is not set: copy .env.example to .env and fill it in")
    return v


def post(url, body, key, tries=5):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            return json.loads(urllib.request.urlopen(req, timeout=180).read())
        except urllib.error.HTTPError as e:
            if (e.code == 429 or e.code >= 500) and a < tries - 1:
                time.sleep(3 * (a + 1))
                continue
            raise SystemExit(f"HTTP {e.code} from {url}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError):
            if a < tries - 1:
                time.sleep(3 * (a + 1))
                continue
            raise


# ------------------------------------------------------------------ EDGAR: the filing is a lookup
def sec_get(url, name=None):
    f = CACHE / name if name else None
    if f and f.exists():
        return f.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": env("SEC_USER_AGENT")})
    data = urllib.request.urlopen(req, timeout=120).read()
    if f:
        CACHE.mkdir(exist_ok=True)
        f.write_bytes(data)
    return data


def latest_filing(ticker, form="10-K"):
    tickers = json.loads(sec_get("https://www.sec.gov/files/company_tickers.json", "company_tickers.json"))
    co = next((v for v in tickers.values() if v["ticker"].upper() == ticker.upper()), None)
    if co is None:
        sys.exit(f"unknown ticker {ticker}")
    cik = co["cik_str"]
    rec = json.loads(sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))["filings"]["recent"]
    i = next((i for i, f in enumerate(rec["form"]) if f == form), None)
    if i is None:
        sys.exit(f"no {form} in {ticker}'s recent filings")
    acc = rec["accessionNumber"][i]
    return dict(company=co["title"], ticker=ticker.upper(), form=form, key=acc, period=rec["reportDate"][i],
                filed=rec["filingDate"][i],
                url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{rec['primaryDocument'][i]}")


def filing_text(fil):
    raw = sec_get(fil["url"], f"{fil['key']}.html").decode("utf-8", "ignore")
    raw = re.sub(r"(?is)<ix:header>.*?</ix:header>|<script.*?</script>|<style.*?</style>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).replace("\xa0", " ")


# ------------------------------------------------------------------ first stage: BM25, optional e5, fused with RRF
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def bm25_rank(q, texts):
    docs = [tok(t) for t in texts]
    avg, n = sum(map(len, docs)) / len(docs), len(docs)
    df = Counter(w for d in docs for w in set(d))
    scores = []
    for d in docs:
        tf, s = Counter(d), 0.0
        for w in tok(q):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(d) / avg))
        scores.append(s)
    return sorted(range(n), key=lambda i: -scores[i])


def e5_rank(q, texts):
    """intfloat/multilingual-e5-base, mean pooled; None when torch or transformers is missing."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    m = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(dev).eval()

    def emb(xs):
        out = []
        for i in range(0, len(xs), 16):
            b = tk(xs[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = m(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1)
            out.append(torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=-1))
        return torch.cat(out)
    s = (emb(["query: " + q]) @ emb(["passage: " + t for t in texts]).T)[0].tolist()
    return sorted(range(len(texts)), key=lambda i: -s[i])


def rrf(*rankings, k=60):
    s = Counter()
    for r in rankings:
        for pos, i in enumerate(r):
            s[i] += 1 / (k + pos + 1)
    return [i for i, _ in s.most_common()]


# ------------------------------------------------------------------ Jev picks, code decides
def jev_choice(q, cands, label, key):
    """cands: list of (passage id, text). One call: a Choice over the passages plus a Noul on whether the answer is there."""
    listing = "\n\n".join(f"[{pid}] {txt}" for pid, txt in cands)
    body = {"model": "jev-latest", "state": f"Question: {q}\n\nCandidate passages from {label}:\n\n{listing}",
            "questions": {
                "best": {"type": "choice", "instructions": "Which passage states the information needed to answer the question?",
                         "criteria": {pid: f"passage {pid}" for pid, _ in cands}},
                "exists": {"type": "noul", "instructions": "Do these passages contain the answer to the question?"}}}
    r = post(JEV_URL, body, key)
    p = r["answers"]["best"]["probabilities"]
    u = r.get("usage", {})
    return dict(ranked=sorted(p, key=lambda k: -p[k]), probs=p, exists=r["answers"]["exists"]["noul"],
                tokens=u.get("input_tokens", 0), cost=float(u.get("cost", 0) or 0))


def llm_answer(q, passages, label, model):
    ps = "\n\n".join(f"[{pid}] {txt}" for pid, txt in passages)
    msgs = [{"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"Passages from {label} ($ in millions unless stated):\n\n{ps}\n\nQuestion: {q}"}]
    if model.startswith("deepseek-"):
        r = post(CHAT_URL["deepseek"], {"model": model, "messages": msgs, "max_tokens": 3000}, env("DEEPSEEK_API_KEY"))
    else:
        r = post(CHAT_URL["openrouter"], {"model": model, "messages": msgs, "max_tokens": 4000, "usage": {"include": True}},
                 env("OPENROUTER_API_KEY"))
    u = r.get("usage", {})
    return (r["choices"][0]["message"].get("content") or "").strip(), u


def answer(ticker, question, form="10-K", model=None, embed=True):
    t0 = time.time()
    or_key = env("OPENROUTER_API_KEY")
    model = model or ("deepseek-flash" if os.environ.get("DEEPSEEK_API_KEY") else "qwen/qwen3.7-flash")
    fil = latest_filing(ticker, form)
    text = filing_text(fil)
    passages = [(f"P{i:03d}", text[s:s + STEP]) for i, s in enumerate(range(0, len(text), STEP))]
    texts = [t for _, t in passages]
    ranking = bm25_rank(question, texts)
    vec = e5_rank(question, texts) if embed else None
    first_stage = "BM25 + e5 (RRF)" if vec else "BM25"
    if vec:
        ranking = rrf(ranking, vec)
    # worded as in the benchmark ("Microsoft's 10-K for fiscal 2026"); fiscal year = year of the period end,
    # which holds for the four filers tested (Apple, Microsoft, Nvidia, Amazon) but not for every company
    name = re.sub(r"\s+(inc|corp|corporation|co|ltd|plc|holdings?)\.?$", "", fil["company"], flags=re.I).title()
    when = f"fiscal {fil['period'][:4]}" if form in ("10-K", "20-F", "40-F") else f"the quarter ended {fil['period']}"
    label = f"{name}'s {form} for {when}"
    jev_calls = []
    for rnd in range(2):                                   # answer not in these 30: try the next 30
        cands = [passages[i] for i in ranking[rnd * K:(rnd + 1) * K]]
        if not cands:
            break
        jev_calls.append(jev_choice(question, cands, label, or_key))
        if jev_calls[-1]["exists"] >= 0.5:
            break
    pick = jev_calls[-1]
    top2 = pick["ranked"][:2]
    by_id = dict(passages)
    ans, usage = llm_answer(question, [(pid, by_id[pid]) for pid in top2], label, model)
    return dict(question=question, filing=fil, first_stage=first_stage, passages_in_filing=len(passages),
                jev=dict(calls=len(jev_calls), exists=pick["exists"], top2={pid: round(pick["probs"][pid], 3) for pid in top2},
                         tokens=sum(c["tokens"] for c in jev_calls), cost=sum(c["cost"] for c in jev_calls)),
                llm=dict(model=model, tokens_in=usage.get("prompt_tokens", 0), tokens_out=usage.get("completion_tokens", 0),
                         cost=float(usage.get("cost", 0) or 0)),
                passages={pid: by_id[pid] for pid in top2}, answer=ans, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description="Answer a question from a company's latest SEC filing with Jev RAG.")
    ap.add_argument("ticker")
    ap.add_argument("question")
    ap.add_argument("--form", default="10-K", help="10-K (default), 10-Q, 20-F ...")
    ap.add_argument("--llm", default=None, help="OpenRouter model id, or deepseek-flash for DeepSeek's API")
    ap.add_argument("--no-embed", action="store_true", help="BM25 only, no vector search")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args()
    r = answer(a.ticker, a.question, a.form, a.llm, not a.no_embed)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    f, j, l = r["filing"], r["jev"], r["llm"]
    print(f"{f['company']} {f['form']}, period {f['period']}, filed {f['filed']} ({r['passages_in_filing']} passages)")
    print(f"first stage {r['first_stage']}; Jev: {j['calls']} call(s), answer there p={j['exists']:.2f}, "
          f"picked {', '.join(f'{k} ({v:.2f})' for k, v in j['top2'].items())}")
    print(f"\n{r['answer']}\n")
    print(f"LLM {l['model']}: {l['tokens_in']:,} tokens read | Jev ${j['cost']:.5f} + LLM ${l['cost']:.5f} | {r['seconds']}s")


if __name__ == "__main__":
    main()
