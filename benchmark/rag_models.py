"""Same Agentic RAG vs Jev RAG test with three more LLMs through OpenRouter: Gemini 2.5 Flash-Lite, GPT-5 nano, Qwen 3.7 Flash.

Questions, filings and grading are those of rag_e2e.py (Apple, 20) and rag_multi.py (Microsoft, Nvidia, Amazon, 10 each).
Jev's picks do not depend on the LLM, so the Jev RAG side reuses the two passages Jev chose in those runs and only the
answer is rewritten by each model. The agentic side is rerun in full with each model as agent and filing reader, today's
date in the prompt as before. Provider defaults for reasoning, same as DeepSeek Flash ran with its default thinking.

Every call and finished question is appended to e2e_log_models/ as it happens; a rerun skips finished work. OpenRouter
spend for this run is capped at CAP dollars (usage.cost from each response). Key from the environment, never printed.

Gemini 2.5 Flash-Lite ran too and is kept in the logs and the summary; the write-up leaves it out because it asked
the user back or skipped its tools on about half of the agentic questions (see README).

    python rag_models.py             everything
    python rag_models.py --summary   table from the logs
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rag_multi as mu                    # companies, gold patterns, HOJE, nota (also imports rag_e2e and retrieval_bench)
e2e, rb = mu.e2e, mu.rb

HERE = Path(__file__).parent
LOG = HERE / "e2e_log_models"
LOG.mkdir(exist_ok=True)
CALLS, RESULTS = LOG / "calls.jsonl", LOG / "results.jsonl"
CAP = 1.20
MODELOS = {"gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite", "gpt-5-nano": "openai/gpt-5-nano",
           "qwen3.7-flash": "qwen/qwen3.7-flash"}
TRAVA = threading.Lock()


class SemCredito(Exception):
    pass


def anexa(arq, rec):
    with TRAVA, arq.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def gasto():
    if not CALLS.exists():
        return 0.0
    with TRAVA:
        return sum(json.loads(l).get("cost", 0) for l in CALLS.open(encoding="utf-8"))


def chat(modelo, messages, rotulo, tools=None, max_tokens=6000):
    if gasto() > CAP:
        raise SemCredito(f"OpenRouter spend passed the ${CAP} cap")
    body = {"model": MODELOS[modelo], "messages": messages, "max_tokens": max_tokens, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
    t0 = time.time()
    for tentativa in range(5):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {rb.OR_KEY}"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            if "choices" in r:
                break
            raise RuntimeError(str(r.get("error", r))[:300])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as e:
            if getattr(e, "code", 0) == 402:
                raise SemCredito("OpenRouter says the balance is exhausted (402)")
            if tentativa == 4:
                raise
            time.sleep(4 * (tentativa + 1))
    u = r.get("usage", {})
    custo = float(u.get("cost", 0) or 0)
    anexa(CALLS, dict(t=time.time(), modelo=modelo, rotulo=rotulo, seg=round(time.time() - t0, 2),
                      tin=u.get("prompt_tokens", 0), tout=u.get("completion_tokens", 0), cost=custo))
    return r["choices"][0]["message"], u, custo


def agentic(modelo, tk, razao, q, padroes):
    t0 = time.time()
    msgs = [{"role": "system", "content": e2e.SIS_AGENTE + mu.HOJE}, {"role": "user", "content": f"Question about {razao} (ticker {tk}): {q}"}]
    rec = dict(llm_calls=0, llm_tokens_in=0, llm_tokens_out=0, cost=0.0, reads=0, trace=[])

    def conta(u, c):
        rec["llm_calls"] += 1
        rec["llm_tokens_in"] += u.get("prompt_tokens", 0)
        rec["llm_tokens_out"] += u.get("completion_tokens", 0)
        rec["cost"] += c
    resposta = "(no final answer in 8 turns)"
    for _ in range(8):
        m, u, c = chat(modelo, msgs, "agent_turn", tools=e2e.TOOLS)
        conta(u, c)
        msgs.append({k: v for k, v in m.items() if v is not None})
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
                    rm, ru, rc = chat(modelo, [{"role": "system", "content": e2e.SIS_LEITOR},
                                               {"role": "user", "content": f"FILING {key}:\n{texto}\n\nQUESTION: {args.get('question', q)}"}],
                                      "read_filing")
                    conta(ru, rc)
                    out = (rm.get("content") or "").strip() or "(the reader returned nothing)"
                    rec["trace"].append(f"read {key}: {args.get('question')}")
            elif nome == "calculator":
                out = e2e.calc(args.get("expression", ""))
                rec["trace"].append(f"calc {args.get('expression')} = {out}")
            else:
                out = "unknown tool"
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": out})
    rec.update(answer=resposta, correct=mu.nota(padroes, resposta), seconds=time.time() - t0)
    return rec


def perguntas():
    """All 50 questions with their gold patterns and the two passages Jev picked for each (from the earlier runs)."""
    out = []
    ap = {json.loads(l)["qi"]: json.loads(l) for l in (HERE / "e2e_log/results.jsonl").open(encoding="utf-8")
          if json.loads(l)["metodo"] == "jev_rag_deepseek"}
    for qi, (q, _) in enumerate(rb.QS):
        pas = {p["id"]: p["text"] for p in rb.P}
        out.append(dict(tk="AAPL", qi=qi, q=q, razao="Apple Inc.", nome="Apple", fy=2025, padroes=e2e.GRADE[qi + 1],
                        top2=[(i, pas[i]) for i in ap[qi + 1]["passages"]]))
    mj = {(r["tk"], r["qi"]): r for r in map(json.loads, (HERE / "e2e_log_multi/results.jsonl").open(encoding="utf-8"))
          if r["metodo"] == "jev_rag_deepseek"}
    for tk, emp in mu.EMPRESAS.items():
        texto = e2e.texto_filing(emp["key"])
        pas = {f"P{i:03d}": texto[s:s + mu.STEP] for i, s in enumerate(range(0, len(texto), mu.STEP))}
        for qi, (q, _, padroes) in enumerate(emp["qs"]):
            out.append(dict(tk=tk, qi=qi, q=q, razao=emp["razao"], nome=emp["nome"], fy=emp["fy"], padroes=padroes,
                            top2=[(i, pas[i]) for i in mj[(tk, qi)]["passages"]]))
    return out


def roda_modelo(modelo, qs, feitos):
    for x in qs:
        try:
            if (modelo, x["tk"], x["qi"], "jev_rag") not in feitos:
                ps = "\n\n".join(f"[{i}] {t}" for i, t in x["top2"])
                msgs = [{"role": "system", "content": e2e.SIS_RESPOSTA},
                        {"role": "user", "content": f"Passages from {x['nome']}'s 10-K for fiscal {x['fy']} ($ in millions unless stated):\n\n{ps}\n\nQuestion: {x['q']}"}]
                t0 = time.time()
                m, u, c = chat(modelo, msgs, "answer", max_tokens=4000)
                ans = (m.get("content") or "").strip()
                anexa(RESULTS, dict(modelo=modelo, tk=x["tk"], qi=x["qi"], question=x["q"], metodo="jev_rag", answer=ans,
                                    correct=mu.nota(x["padroes"], ans), llm_calls=1, llm_tokens_in=u.get("prompt_tokens", 0),
                                    llm_tokens_out=u.get("completion_tokens", 0), cost=c, seconds=time.time() - t0))
            if (modelo, x["tk"], x["qi"], "agentic_rag") not in feitos:
                rec = agentic(modelo, x["tk"], x["razao"], x["q"], x["padroes"])
                anexa(RESULTS, dict(rec, modelo=modelo, tk=x["tk"], qi=x["qi"], question=x["q"], metodo="agentic_rag"))
                print(f"  {modelo:22s} {x['tk']} q{x['qi'] + 1:<2} agentic {'RIGHT' if rec['correct'] else 'wrong'} "
                      f"{rec['llm_calls']} calls {rec['llm_tokens_in']:,} tok ${rec['cost']:.4f} | spent ${gasto():.3f}", flush=True)
        except SemCredito as e:
            print(f"STOPPED {modelo}: {e}", flush=True)
            return
        except Exception as e:                            # one bad question should not kill the model's run
            print(f"  {modelo} {x['tk']} q{x['qi'] + 1} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)


# read by hand on 28 Sep 2026: (modelo, tk, qi, metodo) -> (correct, reason), where the automatic check got it wrong
MANUAL = {
    ("gemini-2.5-flash-lite", "AAPL", 10, "jev_rag"): (True, "term debt principal $91,281m, a true figure in the filing"),
    ("gpt-5-nano", "AAPL", 10, "jev_rag"): (True, "term debt principal $91,281m, a true figure in the filing"),
    ("gpt-5-nano", "AAPL", 9, "jev_rag"): (True, "402 million shares repurchased: answers how much stock"),
    ("gemini-2.5-flash-lite", "NVDA", 8, "jev_rag"): (True, "Blackwell demand, NVLink ramp: correct drivers, other wording"),
    ("gpt-5-nano", "NVDA", 8, "jev_rag"): (True, "Blackwell demand, NVLink ramp: correct drivers, other wording"),
    ("qwen3.7-flash", "NVDA", 8, "jev_rag"): (True, "Blackwell demand, NVLink ramp: correct drivers, other wording"),
    ("gpt-5-nano", "MSFT", 7, "agentic_rag"): (True, "cash-flow repurchases $22,271m for FY2026, same figure accepted for DeepSeek Flash"),
    ("gpt-5-nano", "AMZN", 6, "agentic_rag"): (False, "right reason, wrong year: 2024 growth from the prior filing"),
}


def resumo():
    res = [json.loads(l) for l in RESULTS.open(encoding="utf-8")] if RESULTS.exists() else []
    for r in res:
        k = (r["modelo"], r["tk"], r["qi"], r["metodo"])
        if k in MANUAL:
            r["correct"] = MANUAL[k][0]
    tab = {}
    print(f"\n{'model':22s} {'pipeline':12s} right   AAPL MSFT NVDA AMZN  calls  tokens read  $ total  s/q")
    for modelo in MODELOS:
        for m in ("agentic_rag", "jev_rag"):
            x = [r for r in res if r["modelo"] == modelo and r["metodo"] == m]
            if not x:
                continue
            n = len(x)
            por = {tk: sum(r["correct"] for r in x if r["tk"] == tk) for tk in ("AAPL", "MSFT", "NVDA", "AMZN")}
            tab[f"{modelo}:{m}"] = t = dict(n=n, right=sum(r["correct"] for r in x), by_company=por,
                                            llm_calls=sum(r["llm_calls"] for r in x) / n,
                                            tokens_read=sum(r["llm_tokens_in"] for r in x) / n, cost=sum(r["cost"] for r in x),
                                            seconds=sum(r["seconds"] for r in x) / n)
            print(f"{modelo:22s} {m:12s} {t['right']:>2}/{n:<3} {por['AAPL']:>4} {por['MSFT']:>4} {por['NVDA']:>4} {por['AMZN']:>4}  "
                  f"{t['llm_calls']:5.1f} {t['tokens_read']:>11,.0f}  {t['cost']:7.4f} {t['seconds']:5.1f}")
    print(f"\nOpenRouter spent in this run ${gasto():.4f} (cap ${CAP})")
    (HERE / "rag_models_summary.json").write_text(json.dumps(tab, indent=1))


def main():
    if "--summary" in sys.argv:
        return resumo()
    qs = perguntas()
    feitos = {(r["modelo"], r["tk"], r["qi"], r["metodo"]) for r in map(json.loads, RESULTS.open(encoding="utf-8"))} if RESULTS.exists() else set()
    print(f"{len(qs)} questions x {len(MODELOS)} models; spent so far ${gasto():.4f} (cap ${CAP})", flush=True)
    with ThreadPoolExecutor(len(MODELOS)) as ex:
        list(ex.map(lambda m: roda_modelo(m, qs, feitos), MODELOS))
    resumo()


if __name__ == "__main__":
    main()
