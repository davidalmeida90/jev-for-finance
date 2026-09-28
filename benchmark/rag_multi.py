"""Agentic RAG vs Jev RAG on three more large caps, plus a passage-order check for Jev.

Microsoft (10-K for fiscal 2026, year ended June), Nvidia (fiscal 2026, ended January) and Amazon (fiscal 2025, ended
December), ten analyst questions each, answers verified verbatim in each filing. Same pipelines as rag_e2e.py (Apple):
  retrieval   BM25, e5, hybrid (RRF), MiniLM and bge rerankers on the hybrid top 30 (local, free), Jev one Noul pair
              per candidate, Jev Choice over the 30 in one call
  order check Jev Choice again with the 30 candidates in two shuffled orders, on these 30 questions and Apple's 20
  end to end  Agentic RAG (DeepSeek agent, EDGAR search, whole-filing reads, calculator) against Jev RAG (code finds the
              latest 10-K, hybrid top 30, Jev Choice plus existence check, LLM reads the top 2), the LLM step run with
              DeepSeek Flash and with qwen3.5:4b on the local GPU
Skipped here to save credit: DeepSeek ranking and Jev over every passage (both measured on Apple only).

Every call and every finished question is appended to e2e_log_multi/ as it happens; a rerun skips finished work.
DeepSeek spend for this run is capped at CAP dollars. Keys from the environment (see .env.example), never printed.

    python rag_multi.py             everything
    python rag_multi.py --summary   tables from the logs, no calls
"""
import json
import math
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rag_e2e as e2e                  # DeepSeek wrapper with spend log, EDGAR, calculator, agent tools, local Qwen
import retrieval_bench as rb           # Apple passages and questions, RRF, OpenRouter post, Jev key

HERE = Path(__file__).parent
LOG = HERE / "e2e_log_multi"
LOG.mkdir(exist_ok=True)
e2e.CALLS = LOG / "calls.jsonl"        # this run keeps its own spend log and cap
e2e.CAP = 0.30
RETR, RESULTS = LOG / "retrieval.jsonl", LOG / "results.jsonl"
JEV_URL = "https://openrouter.ai/api/v1/systemone"
# First run (set aside in *_v1_*.jsonl): without the date the agent assumed Microsoft's latest year was fiscal 2025
# and never searched 2026. Harnesses normally give an agent the date, so it gets it here, for all 50 questions.
HOJE = " Today's date is 27 September 2026."
STEP, K = 1500, 30

EMPRESAS = {
    "MSFT": dict(razao="Microsoft Corporation", nome="Microsoft", fy=2026, key="0001193125-26-323660", qs=[
        ("What was Microsoft's total revenue in its latest fiscal year?", "331,839", [[r"331\.?8"]]),
        ("What was Microsoft's bottom line in its latest fiscal year?", "133,749", [[r"133\.?7"]]),
        ("What were Microsoft's diluted earnings per share?", "17.95", [[r"17\.95"]]),
        ("How much does Microsoft spend on R&D?", "35,562", [[r"35\.?56|35\.6 ?billion"]]),
        ("How big is the Microsoft Cloud business?", "214.4 billion", [[r"214\.4"]]),
        ("How fast did Azure grow, and why?", "Azure and other cloud services revenue grew 41%", [[r"41", r"demand"]]),
        ("How many people work at Microsoft?", "223,000", [[r"223 ?000|223k|223 thousand"]]),
        ("How much stock did Microsoft buy back in its latest fiscal year?", "16.7 billion", [[r"16\.7"]]),
        ("Why did Microsoft's tax rate go up?", "The increase in our effective tax rate was primarily due to changes in the mix", [[r"mix"]]),
        ("What was Microsoft's operating income?", "155,237", [[r"155\.?2"]]),
    ]),
    "NVDA": dict(razao="NVIDIA Corporation", nome="Nvidia", fy=2026, key="0001045810-26-000021", qs=[
        ("What was Nvidia's revenue in its latest fiscal year?", "215,938", [[r"215\.?9"]]),
        ("How much did Nvidia earn in its latest fiscal year?", "120,067", [[r"120\.?0[67]|120\.1"]]),
        ("What were Nvidia's diluted earnings per share?", "4.90", [[r"4\.90?\b"]]),
        ("How much does Nvidia spend on R&D?", "18,497", [[r"18497|18\.49|18\.5 ?billion"]]),
        ("How big is Nvidia's data center business?", "193,737", [[r"193\.?7"]]),
        ("Why did Nvidia's gross margin fall?", "Blackwell full-scale datacenter solutions", [[r"blackwell"], [r"h20"]]),
        ("How many employees does Nvidia have?", "42,000 employees", [[r"42 ?000|42k|42 thousand"]]),
        ("How much stock did Nvidia repurchase in its latest fiscal year?", "40.4 billion", [[r"40\.4"]]),
        ("What drove Nvidia's data center growth?", "The strong year-on-year growth was driven by the major platform shifts", [[r"accelerated computing", r"\bai\b"]]),
        ("What was Nvidia's tax rate in its latest fiscal year?", "15.1%", [[r"15\.1"]]),
    ]),
    "AMZN": dict(razao="Amazon.com, Inc.", nome="Amazon", fy=2025, key="0001018724-26-000004", qs=[
        ("What were Amazon's total net sales in its latest fiscal year?", "716,924", [[r"716\.?9"]]),
        ("What was Amazon's net income?", "77,670", [[r"77\.?6[67]|77\.7"]]),
        ("What were Amazon's diluted earnings per share?", "7.17", [[r"7\.17"]]),
        ("How much did AWS sell in Amazon's latest fiscal year?", "128,725", [[r"128\.?7"]]),
        ("How profitable is AWS?", "45,606", [[r"45\.?6"]]),
        ("How many people does Amazon employ?", "1,576,000", [[r"1576000|1\.576|1\.58 ?million"]]),
        ("Why did Amazon's North America sales grow?", "North America sales increased 10% in 2025", [[r"unit sales|third-party|advertising|subscription"]]),
        ("Why did AWS sales grow?", "AWS sales increased 20% in 2025", [[r"usage"]]),
        ("What was Amazon's free cash flow?", "11,194", [[r"11\.?19|11\.2 ?billion"]]),
        ("What was Amazon's operating income?", "79,975", [[r"79\.?9[78]|80\.0 ?billion|\$80 ?billion"]]),
    ]),
}


def nota(padroes, ans):
    a = re.sub(r"(?<=\d),(?=\d{3})", "", ans.lower())
    return any(all(re.search(p, a) for p in alt) for alt in padroes)


def anexa(arq, rec):
    with arq.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def lidos(arq):
    return [json.loads(l) for l in arq.open(encoding="utf-8")] if arq.exists() else []


# ------------------------------------------------------------------ documents and first stage
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, textos):
        self.docs = [tok(t) for t in textos]
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        self.df = Counter(w for d in self.docs for w in set(d))

    def scores(self, q):
        n, out = len(self.docs), []
        for d in self.docs:
            tf, s = Counter(d), 0.0
            for w in tok(q):
                if w in tf:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(d) / self.avg))
            out.append(s)
        return out


def ordem(scores):
    return sorted(range(len(scores)), key=lambda i: -scores[i])


class Locais:
    """e5 embeddings and the two cross-encoders, loaded once on the GPU."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.dev = torch, ("cuda" if torch.cuda.is_available() else "cpu")
        self.tk = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        self.e5 = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(self.dev).eval()
        self.ce = {}
        for nome, chave in (("cross-encoder/ms-marco-MiniLM-L-6-v2", "minilm"), ("BAAI/bge-reranker-base", "bge")):
            self.ce[chave] = (AutoTokenizer.from_pretrained(nome), AutoModelForSequenceClassification.from_pretrained(nome).to(self.dev).eval())

    def emb(self, textos):
        vs = []
        for i in range(0, len(textos), 16):
            b = self.tk(textos[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                h = self.e5(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            vs.append(self.torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1))
        return self.torch.cat(vs)

    def rerank(self, chave, q, textos):
        tk, m = self.ce[chave]
        b = tk([q] * len(textos), textos, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            return m(**b).logits[:, 0].tolist()


# ------------------------------------------------------------------ Jev (OpenRouter), generic over any filing
def jev_par(q, texto, rotulo):
    body = {"model": "jev-latest", "state": f"Question: {q}\n\nPassage from {rotulo}:\n{texto}",
            "questions": {
                "is_relevant": {"type": "noul", "instructions": "Does this passage address the subject of the question?"},
                "contains_answer_evidence": {"type": "noul",
                    "instructions": "Does this passage state the information needed to answer the question?",
                    "criteria": {"true": "The passage gives the specific figures or facts that answer it",
                                 "false": "The passage is only on a related topic"}}}}
    r = rb.post(JEV_URL, body, rb.OR_KEY)
    return r["answers"]["contains_answer_evidence"]["noul"], float(r.get("usage", {}).get("cost", 0) or 0)


def jev_choice(q, cands, rotulo):
    """cands: list of (passage id, text) in the order Jev should see them."""
    lista = "\n\n".join(f"[{pid}] {txt}" for pid, txt in cands)
    body = {"model": "jev-latest", "state": f"Question: {q}\n\nCandidate passages from {rotulo}:\n\n{lista}",
            "questions": {
                "best": {"type": "choice", "instructions": "Which passage states the information needed to answer the question?",
                         "criteria": {pid: f"passage {pid}" for pid, _ in cands}},
                "exists": {"type": "noul", "instructions": "Do these passages contain the answer to the question?"}}}
    t0 = time.time()
    r = rb.post(JEV_URL, body, rb.OR_KEY)
    pr = r["answers"]["best"]["probabilities"]
    return dict(ids=sorted(pr, key=lambda k: -pr[k]), exists=r["answers"]["exists"]["noul"],
                cost=float(r.get("usage", {}).get("cost", 0) or 0), seconds=time.time() - t0)


def embaralha(cands, semente):
    c = list(cands)
    random.Random(semente).shuffle(c)
    return c


# ------------------------------------------------------------------ agent, as in rag_e2e.agentic but for any company
def agentic(tk, razao, q, padroes):
    t0 = time.time()
    msgs = [{"role": "system", "content": e2e.SIS_AGENTE + HOJE}, {"role": "user", "content": f"Question about {razao} (ticker {tk}): {q}"}]
    rec = dict(llm_calls=0, llm_tokens_in=0, llm_tokens_out=0, ds_cost=0.0, ds_cold=0.0, jev_calls=0, jev_cost=0.0, reads=0, trace=[])

    def conta(u, a, c):
        rec["llm_calls"] += 1
        rec["llm_tokens_in"] += u.get("prompt_tokens", 0)
        rec["llm_tokens_out"] += u.get("completion_tokens", 0)
        rec["ds_cost"] += a
        rec["ds_cold"] += c
    resposta = "(no final answer in 8 turns)"
    for _ in range(8):
        m, u, a, c = e2e.deepseek(msgs, None, "agentic_rag", "agent_turn", tools=e2e.TOOLS)
        conta(u, a, c)
        msgs.append(m)
        if not m.get("tool_calls"):
            resposta = (m.get("content") or "").strip()
            break
        for tc in m["tool_calls"]:
            nome = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if nome == "search_filings":
                res = e2e.search_filings(args.get("ticker", ""), args.get("form", "10-K"), args.get("fiscal_year"))
                out = json.dumps([{k: x[k] for k in ("key", "form", "filed", "period")} for x in res] if isinstance(res, list) else res)
                rec["trace"].append(f"search {args}")
            elif nome == "read_filing":
                key = args.get("key", "")
                if rec["reads"] >= 3:
                    out = "read limit reached; answer with the evidence you have"
                elif key not in e2e.URLS and not (e2e.FILINGS / f"{key}.txt").exists():
                    out = "unknown key; call search_filings first"
                else:
                    rec["reads"] += 1
                    texto = e2e.texto_filing(key)
                    rm, ru, ra, rc = e2e.deepseek([{"role": "system", "content": e2e.SIS_LEITOR},
                                                   {"role": "user", "content": f"FILING {key}:\n{texto}\n\nQUESTION: {args.get('question', q)}"}],
                                                  None, "agentic_rag", "read_filing")
                    conta(ru, ra, rc)
                    out = (rm.get("content") or "").strip() or "(the reader returned nothing)"
                    rec["trace"].append(f"read {key}: {args.get('question')}")
            elif nome == "calculator":
                out = e2e.calc(args.get("expression", ""))
                rec["trace"].append(f"calc {args.get('expression')} = {out}")
            else:
                out = "unknown tool"
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    rec.update(answer=resposta, correct=nota(padroes, resposta), seconds=time.time() - t0)
    return rec


# ------------------------------------------------------------------ run
def main():
    if "--summary" in sys.argv:
        return resumo()
    print(f"DeepSeek balance ${e2e.saldo():.2f}; spent in this run's log ${e2e.gasto():.4f} (cap ${e2e.CAP})", flush=True)
    feitos_r = {(r["tk"], r["qi"]) for r in lidos(RETR)}
    feitos_e = {(r["tk"], r["qi"], r["metodo"]) for r in lidos(RESULTS)}
    loc = Locais()

    # --- Apple: order check only (its other results come from retrieval_bench.py and rag_e2e.py)
    loc_apple = json.loads((rb.CACHE / "local.json").read_text())
    choice_apple = json.loads((rb.CACHE / "jev_choice.json").read_text())
    rot_apple = "Apple's 10-K for fiscal 2025"

    def ordem_apple(qi):
        q = rb.QS[qi][0]
        top = rb.rrf(rb.rank_of(rb.bm25(q)), rb.rank_of(loc_apple["e5"][qi]))[:K]
        cands = [(rb.P[i]["id"], rb.P[i]["text"]) for i in top]
        rec = dict(tk="AAPL", qi=qi, q=q, gold=sorted(rb.GOLD[qi]), orig=choice_apple[qi]["ids"][:3])
        for s in (1, 2):
            c = jev_choice(q, embaralha(cands, 1000 * s + qi), rot_apple)
            rec[f"shuf{s}"], rec[f"shuf{s}_cost"] = c["ids"][:3], c["cost"]
        return rec
    pend = [qi for qi in range(len(rb.QS)) if ("AAPL", qi) not in feitos_r]
    with ThreadPoolExecutor(4) as ex:
        for rec in ex.map(ordem_apple, pend):
            anexa(RETR, rec)
    print(f"Apple order check done ({len(pend)} new)", flush=True)

    try:
        for qi, (q, _) in enumerate(rb.QS):
            if ("AAPL", qi, "agentic_rag") not in feitos_e:
                padroes = e2e.GRADE[qi + 1]
                rec = agentic("AAPL", "Apple Inc.", q, padroes)
                anexa(RESULTS, dict(rec, tk="AAPL", qi=qi, question=q, metodo="agentic_rag"))
                print(f"  AAPL q{qi + 1} agentic: {'RIGHT' if rec['correct'] else 'wrong'}, {rec['llm_calls']} calls, "
                      f"{rec['llm_tokens_in']:,} tokens, ${rec['ds_cost']:.4f}; spent so far ${e2e.gasto():.4f}", flush=True)
        for tk, emp in EMPRESAS.items():
            texto = e2e.texto_filing(emp["key"])
            fil = e2e.search_filings(tk, "10-K")[0]          # the Jev side's lookup: latest 10-K
            assert fil["key"] == emp["key"], (tk, fil)
            P = [{"id": f"P{i:03d}", "text": texto[s:s + STEP]} for i, s in enumerate(range(0, len(texto), STEP))]
            textos = [p["text"] for p in P]
            rotulo = f"{emp['nome']}'s 10-K for fiscal {emp['fy']}"
            bm = BM25(textos)
            pv = loc.emb(["passage: " + t for t in textos])
            print(f"{tk}: {len(P)} passages", flush=True)
            for qi, (q, ouro, padroes) in enumerate(emp["qs"]):
                gold = {p["id"] for p in P if ouro in p["text"]}
                assert gold, (tk, ouro)
                # ---- retrieval
                if (tk, qi) not in feitos_r:
                    r_bm = ordem(bm.scores(q))
                    r_e5 = ordem((loc.emb(["query: " + q]) @ pv.T)[0].tolist())
                    hib = rb.rrf(r_bm, r_e5)
                    top = hib[:K]
                    rec = dict(tk=tk, qi=qi, q=q, gold=sorted(gold), recall30=any(P[i]["id"] in gold for i in top),
                               bm25=[P[i]["id"] for i in r_bm[:3]], e5=[P[i]["id"] for i in r_e5[:3]], hybrid=[P[i]["id"] for i in hib[:3]])
                    for chave in ("minilm", "bge"):
                        sc = loc.rerank(chave, q, [P[i]["text"] for i in top])
                        rec[chave] = [P[top[j]]["id"] for j in ordem(sc)[:3]]
                    with ThreadPoolExecutor(8) as ex:
                        pares = list(ex.map(lambda i: jev_par(q, P[i]["text"], rotulo), top))
                    rec["jev_pairs"] = [P[top[j]]["id"] for j in ordem([p[0] for p in pares])[:3]]
                    rec["jev_pairs_cost"] = sum(p[1] for p in pares)
                    cands = [(P[i]["id"], P[i]["text"]) for i in top]
                    c0 = jev_choice(q, cands, rotulo)
                    rec.update(orig=c0["ids"][:3], orig_exists=c0["exists"], orig_cost=c0["cost"], orig_seconds=c0["seconds"],
                               hib30=[P[i]["id"] for i in hib[:60]])
                    for s in (1, 2):
                        c = jev_choice(q, embaralha(cands, 1000 * s + qi), rotulo)
                        rec[f"shuf{s}"], rec[f"shuf{s}_cost"] = c["ids"][:3], c["cost"]
                    anexa(RETR, rec)
                    feitos_r.add((tk, qi))
                    print(f"  {tk} q{qi + 1} retrieval: hybrid {'ok' if rec['hybrid'][0] in gold else '--'}, "
                          f"Jev {'ok' if rec['orig'][0] in gold else '--'}", flush=True)
                rr = next(r for r in lidos(RETR) if r["tk"] == tk and r["qi"] == qi)
                por_id = {p["id"]: p["text"] for p in P}
                # ---- Jev RAG end to end (retrieval call above, answer by DeepSeek and by local Qwen)
                if not {(tk, qi, "jev_rag_deepseek"), (tk, qi, "jev_rag_qwen_local")} <= feitos_e:
                    ids, existe, jcost, jsec, jcalls = rr["orig"], rr["orig_exists"], rr["orig_cost"], rr["orig_seconds"], 1
                    if existe < 0.5:                              # answer not there: the next 30
                        c = jev_choice(q, [(i, por_id[i]) for i in rr["hib30"][K:2 * K]], rotulo)
                        ids, existe, jcost, jsec, jcalls = c["ids"], c["exists"], jcost + c["cost"], jsec + c["seconds"], 2
                    top2 = ids[:2]
                    ps = "\n\n".join(f"[{i}] {por_id[i]}" for i in top2)
                    msgs = [{"role": "system", "content": e2e.SIS_RESPOSTA},
                            {"role": "user", "content": f"Passages from {emp['nome']}'s 10-K for fiscal {emp['fy']} ($ in millions unless stated):\n\n{ps}\n\nQuestion: {q}"}]
                    base = dict(tk=tk, qi=qi, question=q, passages=top2, gold_in_top2=any(i in gold for i in top2), answer_there=existe,
                                jev_calls=jcalls, jev_cost=jcost, reads=0)
                    t1 = time.time()
                    m, u, a, c = e2e.deepseek(msgs, None, "jev_rag_deepseek", "answer", max_tokens=3000)
                    ans = (m.get("content") or "").strip()
                    anexa(RESULTS, dict(base, metodo="jev_rag_deepseek", answer=ans, correct=nota(padroes, ans), llm_calls=1,
                                        llm_tokens_in=u.get("prompt_tokens", 0), llm_tokens_out=u.get("completion_tokens", 0),
                                        ds_cost=a, ds_cold=c, seconds=jsec + time.time() - t1))
                    t1 = time.time()
                    ans, tin, tout = e2e.qwen_local(msgs)
                    anexa(RESULTS, dict(base, metodo="jev_rag_qwen_local", answer=ans, correct=nota(padroes, ans), llm_calls=1,
                                        llm_tokens_in=tin, llm_tokens_out=tout, ds_cost=0.0, ds_cold=0.0, seconds=jsec + time.time() - t1))
                # ---- Agentic RAG end to end
                if (tk, qi, "agentic_rag") not in feitos_e:
                    rec = agentic(tk, emp["razao"], q, padroes)
                    anexa(RESULTS, dict(rec, tk=tk, qi=qi, question=q, metodo="agentic_rag"))
                    print(f"  {tk} q{qi + 1} agentic: {'RIGHT' if rec['correct'] else 'wrong'}, {rec['llm_calls']} calls, "
                          f"{rec['llm_tokens_in']:,} tokens, ${rec['ds_cost']:.4f}; spent so far ${e2e.gasto():.4f}", flush=True)
    except e2e.SemCredito as err:
        print("STOPPED:", err)
    resumo()
    print(f"DeepSeek balance now ${e2e.saldo():.2f} (the balance page can lag a few minutes)")


# ------------------------------------------------------------------ summary
# read by hand on 27 Sep 2026: (tk, qi, metodo) -> (correct, reason), where the automatic check got it wrong
MANUAL = {
    ("MSFT", 7, "agentic_rag"): (True, "cash-flow buyback $22.3bn plus the program's 36m shares for $16,719m; 16,719 missed by the regex"),
    ("NVDA", 8, "jev_rag_deepseek"): (True, "compute +59% on Blackwell demand, networking +142%: correct, other wording"),
    ("NVDA", 9, "jev_rag_qwen_local"): (False, "headline is the prior year's 13.3%"),
    ("AMZN", 6, "agentic_rag"): (False, "right reason, wrong year: describes 2024 growth from the prior filing"),
    ("AMZN", 7, "agentic_rag"): (False, "right reason, wrong year: AWS +19% in 2024"),
    ("AMZN", 2, "jev_rag_qwen_local"): (False, "invents EPS for other years ($7.19, $7.29 for 2023)"),
}


def resumo():
    ret = lidos(RETR)
    print("\nRETRIEVAL, correct passage first (new companies, 10 questions each)")
    metodos = ["bm25", "e5", "hybrid", "minilm", "bge", "jev_pairs", "orig", "shuf1", "shuf2"]
    nomes = ["BM25", "e5", "hybrid", "MiniLM", "bge", "Jev pairs", "Jev Choice", "Choice, shuffle 1", "Choice, shuffle 2"]
    for tk in list(EMPRESAS) + ["ALL3"]:
        rs = [r for r in ret if r["tk"] != "AAPL" and (tk == "ALL3" or r["tk"] == tk)]
        if not rs:
            continue
        linha = "  ".join(f"{n} {sum(r[m][0] in r['gold'] for r in rs)}/{len(rs)}" for m, n in zip(metodos, nomes) if m in rs[0])
        print(f"  {tk:5s} {linha}   recall@30 {sum(r['recall30'] for r in rs)}/{len(rs)}")
    ap = [r for r in ret if r["tk"] == "AAPL"]
    todos = [r for r in ret]
    if ap:
        print(f"\nORDER CHECK, Jev Choice top pick correct: Apple orig {sum(r['orig'][0] in r['gold'] for r in ap)}/{len(ap)}, "
              f"shuffle 1 {sum(r['shuf1'][0] in r['gold'] for r in ap)}/{len(ap)}, shuffle 2 {sum(r['shuf2'][0] in r['gold'] for r in ap)}/{len(ap)}")
    if todos:
        mesma = sum(r["orig"][0] == r["shuf1"][0] == r["shuf2"][0] for r in todos)
        print(f"  all {len(todos)} questions: same top pick in all three orders {mesma}/{len(todos)}; correct in all three "
              f"{sum(all(r[k][0] in r['gold'] for k in ('orig', 'shuf1', 'shuf2')) for r in todos)}/{len(todos)}")
    res = lidos(RESULTS)
    for r in res:
        if (r["tk"], r["qi"], r["metodo"]) in MANUAL:
            r["correct"] = MANUAL[(r["tk"], r["qi"], r["metodo"])][0]
    print("\nEND TO END")
    tab = {}
    # Apple's Jev rows come from rag_e2e.py (same pipeline, same day); its agentic rows are the dated rerun above
    apple = [dict(r, tk="AAPL", qi=r["qi"] - 1) for r in (json.loads(l) for l in (HERE / "e2e_log/results.jsonl").open(encoding="utf-8"))
             if r["metodo"] != "agentic_rag"]
    for r in apple:
        if (r["qi"] + 1, r["metodo"]) in e2e.MANUAL:
            r["correct"] = False
    res = res + apple
    for tk in ["AAPL"] + list(EMPRESAS) + ["ALL4"]:
        for m in ("agentic_rag", "jev_rag_deepseek", "jev_rag_qwen_local"):
            x = [r for r in res if r["metodo"] == m and (tk == "ALL4" or r["tk"] == tk)]
            if not x:
                continue
            n = len(x)
            tab[f"{tk}:{m}"] = t = dict(n=n, right=sum(r["correct"] for r in x), llm_calls=sum(r["llm_calls"] for r in x) / n,
                                        tokens_read=sum(r["llm_tokens_in"] for r in x) / n, ds_cost=sum(r["ds_cost"] for r in x),
                                        cold_per_q=sum(r["ds_cold"] for r in x) / n, jev_cost=sum(r["jev_cost"] for r in x),
                                        seconds=sum(r["seconds"] for r in x) / n)
            print(f"  {tk:5s} {m:20s} {t['right']:>2}/{n:<3} calls {t['llm_calls']:.1f}  tokens {t['tokens_read']:>9,.0f}  "
                  f"DS ${t['ds_cost']:.4f}  uncached/q ${t['cold_per_q']:.5f}  Jev ${t['jev_cost']:.4f}  {t['seconds']:.1f}s")
    jev_total = sum(r.get("jev_pairs_cost", 0) + r.get("orig_cost", 0) + r.get("shuf1_cost", 0) + r.get("shuf2_cost", 0) for r in ret)
    print(f"\nDeepSeek spent in this run ${e2e.gasto():.4f} (cap ${e2e.CAP}); Jev spent in this run ${jev_total:.3f}")
    (HERE / "rag_multi_summary.json").write_text(json.dumps(tab, indent=1))


if __name__ == "__main__":
    main()
