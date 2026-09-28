"""Which retrieval set-up finds the right 10-K passage? Apple FY2025 10-K, 20 analyst questions.

Gold: the passages that contain the exact answer string (a figure or a sentence), found automatically.
Passages: 145 blocks of 1,500 characters of the plain-text 10-K.

First stage over all 145 passages:  BM25, e5 embeddings (intfloat/multilingual-e5-base, local), hybrid (RRF).
Re-rankers on the hybrid top 30:    MiniLM cross-encoder and bge-reranker-base (local, free), Jev pointwise Noul,
                                    Jev Choice listwise (one call), DeepSeek Flash listwise (one call).
Jev as the similarity metric:       one Noul pair per passage over all 145, no first stage (the small-corpus regime).
Jev routing rule (TypeSafe cookbook): keep if relevant >= 0.45 and evidence > 0.55; precision and recall of the kept set.

API keys from the environment (see .env.example), never printed. Results cached in bench_cache/ so reruns cost nothing.

    python retrieval_bench.py
"""
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config

HERE = Path(__file__).parent
CACHE = HERE / "bench_cache"
CACHE.mkdir(exist_ok=True)
TXT = CACHE / "filings" / "0000320193-25-000079.txt"      # Apple 10-K for fiscal 2025, plain text as used in the run
T = TXT.read_text(encoding="utf-8").replace("\xa0", " ").replace("�", "'")
STEP, K = 1500, 30
P = [{"id": f"P{i:03d}", "text": T[s:s + STEP]} for i, s in enumerate(range(0, len(T), STEP))]

QS = [
    ("How did Apple's gross margin change from fiscal 2024 to fiscal 2025?", "195,201"),
    ("How much did Apple spend on R&D in fiscal 2025?", "34,550"),
    ("What was Apple's bottom line in fiscal 2025?", "112,010"),
    ("What were diluted earnings per share for fiscal 2025?", "7.46"),
    ("How big was the Services business in fiscal 2025 revenue?", "109,158"),
    ("How much revenue did iPhone bring in during fiscal 2025?", "209,586"),
    ("What were Mac sales in fiscal 2025?", "33,708"),
    ("What was Apple's effective tax rate in 2025?", "15.6%"),
    ("How many people does Apple employ?", "166,000"),
    ("How much stock did Apple buy back in 2025?", "89.3 billion"),
    ("How much term debt does Apple carry?", "90,678"),
    ("What was operating income in fiscal 2025?", "133,050"),
    ("How much did Apple book as its income tax provision in fiscal 2025?", "20,719"),
    ("What was Apple's total revenue for fiscal 2025?", "416,161"),
    ("How did wearables and accessories sales do in fiscal 2025?", "35,686"),
    ("Why did sales in China fall in 2025?", "Greater China net sales decreased"),
    ("Why did the Services gross margin percentage rise?", "Services gross margin percentage increased"),
    ("Which geographic segments does Apple report?", "segments consist of the Americas"),
    ("Why was the 2025 tax rate lower than in 2024?", "effective tax rate for 2025 was lower compared to 2024"),
    ("How much did Apple pay in dividends in 2025?", "15.4 billion"),
]
GOLD = [{p["id"] for p in P if g in p["text"]} for _, g in QS]
assert all(GOLD), [QS[i][1] for i, g in enumerate(GOLD) if not g]


# ------------------------------------------------------------------ first stage
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


DOCS = [tok(p["text"]) for p in P]
AVG = sum(map(len, DOCS)) / len(DOCS)
DF = Counter(w for d in DOCS for w in set(d))


def bm25(q):
    qt, N, out = tok(q), len(DOCS), []
    for d in DOCS:
        tf, s = Counter(d), 0.0
        for w in qt:
            if w in tf:
                idf = math.log(1 + (N - DF[w] + 0.5) / (DF[w] + 0.5))
                s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(d) / AVG))
        out.append(s)
    return out


def local_models():
    f = CACHE / "local.json"
    if f.exists():
        return json.loads(f.read_text())
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"e5": [], "minilm": [], "bge_rr": []}
    tk = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    m = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(dev).eval()

    def emb(texts):
        vs = []
        for i in range(0, len(texts), 16):
            b = tk(texts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = m(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1)
            v = (h * mask).sum(1) / mask.sum(1)
            vs.append(torch.nn.functional.normalize(v, dim=-1))
        return torch.cat(vs)
    pv = emb(["passage: " + p["text"] for p in P])
    for q, _ in QS:
        out["e5"].append((emb(["query: " + q]) @ pv.T)[0].tolist())
    del m
    for name, key in (("cross-encoder/ms-marco-MiniLM-L-6-v2", "minilm"), ("BAAI/bge-reranker-base", "bge_rr")):
        tk2 = AutoTokenizer.from_pretrained(name)
        m2 = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
        for q, _ in QS:
            sc = []
            for i in range(0, len(P), 16):
                b = tk2([q] * len(P[i:i + 16]), [p["text"] for p in P[i:i + 16]], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(dev)
                with torch.no_grad():
                    sc += m2(**b).logits[:, 0].tolist()
            out[key].append(sc)
        del m2
    f.write_text(json.dumps(out))
    return out


def rank_of(scores):
    return sorted(range(len(P)), key=lambda i: -scores[i])


def rrf(*rankings, k=60):
    s = Counter()
    for r in rankings:
        for pos, i in enumerate(r):
            s[i] += 1 / (k + pos + 1)
    return [i for i, _ in s.most_common()]


# ------------------------------------------------------------------ API rerankers
OR_KEY = config.OPENROUTER_API_KEY
DS_KEY = config.DEEPSEEK_API_KEY


def post(url, body, key, tries=5):
    config.need(key, "OPENROUTER_API_KEY" if "openrouter" in url else "DEEPSEEK_API_KEY")
    for a in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            return json.loads(urllib.request.urlopen(req, timeout=180).read())
        except urllib.error.HTTPError as e:
            if (e.code == 429 or e.code >= 500) and a < tries - 1:     # incl. Cloudflare 52x from OpenRouter
                time.sleep(3 * (a + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if a < tries - 1:
                time.sleep(3 * (a + 1)); continue
            raise


def jev_pair(q, p):
    body = {"model": "jev-latest",
            "state": f"Question: {q}\n\nPassage from Apple's 10-K for fiscal 2025:\n{p['text']}",
            "questions": {
                "is_relevant": {"type": "noul", "instructions": "Does this passage address the subject of the question?"},
                "contains_answer_evidence": {"type": "noul",
                    "instructions": "Does this passage state the information needed to answer the question?",
                    "criteria": {"true": "The passage gives the specific figures or facts that answer it",
                                 "false": "The passage is only on a related topic"}}}}
    r = post("https://openrouter.ai/api/v1/systemone", body, OR_KEY)
    a = r["answers"]
    return a["is_relevant"]["noul"], a["contains_answer_evidence"]["noul"], r.get("usage", {})


def jev_all():
    f = CACHE / "jev_pairs.json"
    if f.exists():
        return json.loads(f.read_text())
    out = []
    with ThreadPoolExecutor(8) as ex:
        for qi, (q, _) in enumerate(QS):
            res = list(ex.map(lambda p: jev_pair(q, p), P))
            out.append({"rel": [r[0] for r in res], "ev": [r[1] for r in res],
                        "tokens": sum(r[2].get("input_tokens", 0) for r in res),
                        "cost": sum(float(r[2].get("cost", 0) or 0) for r in res)})
            print(f"  jev pairs q{qi + 1}: {out[-1]['tokens']} tokens, ${out[-1]['cost']:.4f}", flush=True)
    f.write_text(json.dumps(out))
    return out


def lista(cands):
    return "\n\n".join(f"[{P[i]['id']}] {P[i]['text']}" for i in cands)


def jev_choice(q, cands):
    body = {"model": "jev-latest",
            "state": f"Question: {q}\n\nCandidate passages from Apple's 10-K for fiscal 2025:\n\n{lista(cands)}",
            "questions": {
                "best": {"type": "choice", "instructions": "Which passage states the information needed to answer the question?",
                         "criteria": {P[i]["id"]: f"passage {P[i]['id']}" for i in cands}},
                "exists": {"type": "noul", "instructions": "Do these passages contain the answer to the question?"}}}
    r = post("https://openrouter.ai/api/v1/systemone", body, OR_KEY)
    pr = r["answers"]["best"]["probabilities"]
    return sorted(pr, key=lambda k: -pr[k]), r["answers"]["exists"]["noul"], r.get("usage", {})


def deepseek_list(q, cands):
    msg = (f"Question: {q}\n\nCandidate passages from Apple's 10-K for fiscal 2025:\n\n{lista(cands)}\n\n"
           "Return only a JSON list with the IDs of up to 5 passages that best answer the question, most useful first.")
    r = post("https://api.deepseek.com/chat/completions",
             {"model": "deepseek-flash", "messages": [{"role": "user", "content": msg}], "temperature": 0, "max_tokens": 3000},
             DS_KEY)
    ids = re.findall(r"P\d{3}", r["choices"][0]["message"]["content"])
    u = r.get("usage", {})
    return list(dict.fromkeys(ids)), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


# ------------------------------------------------------------------ scoring
def metricas(rankings):
    h1 = h3 = rr = 0.0
    for r, g in zip(rankings, GOLD):
        ids = [P[i]["id"] if isinstance(i, int) else i for i in r]
        pos = next((k for k, x in enumerate(ids) if x in g), None)
        h1 += pos == 0
        h3 += pos is not None and pos < 3
        rr += 1 / (pos + 1) if pos is not None and pos < 10 else 0
    n = len(rankings)
    return h1 / n, h3 / n, rr / n


def main():
    loc = local_models()
    R = {}
    R["BM25"] = [rank_of(bm25(q)) for q, _ in QS]
    R["e5 embeddings"] = [rank_of(s) for s in loc["e5"]]
    R["hybrid (BM25 + e5, RRF)"] = [rrf(a, b) for a, b in zip(R["BM25"], R["e5 embeddings"])]
    short = [r[:K] for r in R["hybrid (BM25 + e5, RRF)"]]
    rec = sum(any(P[i]["id"] in g for i in s) for s, g in zip(short, GOLD)) / len(QS)

    def rerank(scores_per_q):
        return [sorted(s, key=lambda i: -sc[i]) + r[K:] for s, sc, r in zip(short, scores_per_q, R["hybrid (BM25 + e5, RRF)"])]
    R["hybrid top 30 + MiniLM cross-encoder"] = rerank(loc["minilm"])
    R["hybrid top 30 + bge-reranker-base"] = rerank(loc["bge_rr"])
    J = jev_all()
    R["hybrid top 30 + Jev (1 question per passage)"] = rerank([j["ev"] for j in J])
    R["Jev over all 145 passages (no first stage)"] = [rank_of(j["ev"]) for j in J]

    fc = CACHE / "jev_choice.json"
    if fc.exists():
        JC = json.loads(fc.read_text())
    else:
        JC = []
        for (q, _), s in zip(QS, short):
            ids, ex, u = jev_choice(q, s)
            JC.append({"ids": ids, "exists": ex, "tokens": u.get("input_tokens", 0), "cost": float(u.get("cost", 0) or 0)})
        fc.write_text(json.dumps(JC))
    R["hybrid top 30 + Jev Choice (1 call per question)"] = [c["ids"] + [P[i]["id"] for i in r] for c, r in zip(JC, R["hybrid (BM25 + e5, RRF)"])]

    fd = CACHE / "deepseek.json"
    if fd.exists():
        DS = json.loads(fd.read_text())
    else:
        DS = []
        for (q, _), s in zip(QS, short):
            ids, pt, ct = deepseek_list(q, s)
            DS.append({"ids": ids, "in": pt, "out": ct})
        fd.write_text(json.dumps(DS))
    R["hybrid top 30 + DeepSeek Flash (1 call per question)"] = [d["ids"] + [P[i]["id"] for i in r] for d, r in zip(DS, R["hybrid (BM25 + e5, RRF)"])]

    print(f"\n20 questions, 145 passages; hybrid top 30 contains a gold passage for {rec:.0%} of questions\n")
    print(f"{'method':52s} hit@1  hit@3  MRR@10")
    rows = []
    for k, v in R.items():
        h1, h3, mrr = metricas(v)
        rows.append((k, h1, h3, mrr))
        print(f"{k:52s} {h1:5.0%}  {h3:5.0%}  {mrr:6.3f}")
    # routing rule on the shortlist
    kept = gold_kept = 0; found = 0
    for j, s, g in zip(J, short, GOLD):
        keep = [i for i in s if j["rel"][i] >= 0.45 and j["ev"][i] > 0.55]
        kept += len(keep); gold_kept += sum(P[i]["id"] in g for i in keep)
        found += any(P[i]["id"] in g for i in keep)
    print(f"\nJev routing rule on the top 30: kept {kept / len(QS):.1f} passages per question, "
          f"{gold_kept / max(kept, 1):.0%} of kept passages are gold, a gold passage kept for {found / len(QS):.0%} of questions")
    jt, jc = sum(j["tokens"] for j in J), sum(j["cost"] for j in J)
    print(f"Jev pairs: {jt:,} tokens ${jc:.3f} (all 145 x 20); Choice: ${sum(c['cost'] for c in JC):.4f}; "
          f"DeepSeek: {sum(d['in'] for d in DS):,} in + {sum(d['out'] for d in DS):,} out tokens")
    (HERE / "retrieval_bench_results.json").write_text(json.dumps(
        {"rows": rows, "shortlist_recall": rec, "routing": {"kept_per_q": kept / len(QS), "precision": gold_kept / max(kept, 1),
                                                           "questions_with_gold_kept": found / len(QS)}}, indent=1))


if __name__ == "__main__":
    main()
