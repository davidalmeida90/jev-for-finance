"""End to end: Agentic RAG vs Jev RAG on the same 20 questions about Apple's FY2025 10-K.

Agentic RAG  a DeepSeek Flash agent with three tools: search_filings (SEC EDGAR submissions API), read_filing (a
             second DeepSeek call reads the WHOLE filing with one focused question) and a calculator. It plans,
             loops and answers. At most 8 turns and 3 filing reads per question.
Jev RAG      code finds the filing (ticker, form and fiscal year are lookups), keyword + vector search keeps 30
             passages, one Jev call ranks them (Choice) and asks whether the answer is there (Noul; if not, the
             next 30), and an LLM reads the top 2 and writes the answer. Answer step run twice: DeepSeek Flash,
             and qwen3.5:4b on the local GPU through Ollama (free).

Scored automatically against the figures in the filing (GRADE below); every answer is kept for a manual read.
Every API call is appended to e2e_log/calls.jsonl and every finished question to e2e_log/results.jsonl as it
happens, so a run cut short by credit keeps what it did; a rerun skips finished questions.
DeepSeek spend is capped at CAP dollars across the whole log. Keys from the environment (see .env.example), never printed.

    python rag_e2e.py              all 20 questions
    python rag_e2e.py --only 1     one question, to check the plumbing
    python rag_e2e.py --summary    table from the log, no calls
"""
import ast
import datetime as dt
import html
import json
import operator as op
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import config
import retrieval_bench as rb          # passages, questions, BM25, RRF, Jev Choice call, keys

HERE = Path(__file__).parent
LOG = HERE / "e2e_log"
LOG.mkdir(exist_ok=True)
CALLS, RESULTS = LOG / "calls.jsonl", LOG / "results.jsonl"
FILINGS = HERE / "bench_cache" / "filings"
FILINGS.mkdir(exist_ok=True)
CAP = 0.20
SEED = "0000320193-25-000079"          # Apple 10-K for FY2025: same plain text the retrieval benchmark used


def ua():
    return {"User-Agent": config.need(config.SEC_USER_AGENT, "SEC_USER_AGENT")}

# a correct answer carries these figures or facts (digit commas removed, lower case); any one alternative will do
GRADE = {1: [[r"46\.9", r"46\.2"], [r"195\.?2", r"180\.?[67]"]], 2: [[r"34\.?55|34\.6 ?billion"]],
         3: [[r"112\.?0|112 ?billion"]], 4: [[r"7\.46"]], 5: [[r"109\.?[12]"]], 6: [[r"209\.?[56]"]],
         7: [[r"33\.?7"]], 8: [[r"15\.6"]], 9: [[r"166 ?000|166k|166 thousand"]], 10: [[r"89\.3"]],
         11: [[r"90\.?[67]"], [r"12\.?35", r"78\.?3"]], 12: [[r"133\.?[01]"]], 13: [[r"20\.?7"]],
         14: [[r"416\.?[12]"]], 15: [[r"35\.?[67]"]], 16: [[r"iphone", r"decreas|declin|lower|fell|drop"]],
         17: [[r"mix"]], 18: [[r"americas", r"europe", r"greater china", r"japan", r"rest of asia"]],
         19: [[r"state aid"]], 20: [[r"15\.4"]]}


# read by hand on 27 Sep 2026: where the automatic check was too lenient
MANUAL = {(17, "agentic_rag"): "right reason, wrong year: quotes the FY2024 10-K (2023 to 2024)",
          (16, "jev_rag_qwen_local"): "adds Services as a cause; the 10-K names iPhone only"}


def nota(qi, ans):
    a = re.sub(r"(?<=\d),(?=\d{3})", "", ans.lower())
    return any(all(re.search(p, a) for p in alt) for alt in GRADE[qi])


# ------------------------------------------------------------------ log and money
def log_call(**k):
    with CALLS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(k) + "\n")


def log_result(rec):
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  q{rec['qi']:>2} {rec['metodo']:20s} {'RIGHT' if rec['correct'] else 'wrong'}  "
          f"{rec['llm_calls']} LLM calls, {rec['llm_tokens_in']:,} tokens read, ${rec['ds_cost']:.4f}, {rec['seconds']:.0f}s", flush=True)


def gasto():
    if not CALLS.exists():
        return 0.0
    return sum(json.loads(l).get("ds_cost", 0) for l in CALLS.open(encoding="utf-8"))


def precos():
    """DeepSeek Flash, $ per token: cache hit, cache miss, output. Peak (x2): weekdays 01-04 and 06-10 UTC."""
    t = dt.datetime.now(dt.timezone.utc)
    k = 2 if t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10) else 1
    return 0.003e-6 * k, 0.15e-6 * k, 0.6e-6 * k


def custo(u):
    hit, miss, out = precos()
    h = u.get("prompt_cache_hit_tokens", 0)
    m = u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0) - h)
    c = u.get("completion_tokens", 0)
    return h * hit + m * miss + c * out, (h + m) * miss + c * out       # as run, and with no cache at all


def saldo():
    config.need(rb.DS_KEY, "DEEPSEEK_API_KEY")
    r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {rb.DS_KEY}"})
    return float(json.loads(urllib.request.urlopen(r, timeout=30).read())["balance_infos"][0]["total_balance"])


class SemCredito(Exception):
    pass


def deepseek(messages, qi, metodo, tipo, tools=None, max_tokens=4000):
    if gasto() > CAP:
        raise SemCredito(f"DeepSeek spend passed the ${CAP} cap")
    config.need(rb.DS_KEY, "DEEPSEEK_API_KEY")
    body = {"model": "deepseek-flash", "messages": messages, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    t0 = time.time()
    for tentativa in range(4):
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {rb.DS_KEY}"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 402:
                raise SemCredito("DeepSeek says the balance is exhausted (402)")
            if e.code in (429, 500, 502, 503) and tentativa < 3:
                time.sleep(5 * (tentativa + 1))
                continue
            print("DeepSeek error", e.code, e.read()[:400])
            raise
    seg = time.time() - t0
    u = r.get("usage", {})
    a, c = custo(u)
    log_call(t=time.time(), qi=qi, metodo=metodo, tipo=tipo, seg=round(seg, 2), usage=u, ds_cost=a, ds_cold=c)
    return r["choices"][0]["message"], u, a, c


def qwen_local(messages):
    body = {"model": "qwen3.5:4b", "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_ctx": 8192}}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    txt = re.sub(r"(?s)<think>.*?</think>", "", r["message"]["content"]).strip()
    return txt, r.get("prompt_eval_count", 0), r.get("eval_count", 0)


# ------------------------------------------------------------------ EDGAR (free; SEC asks for a User-Agent with an email)
URLS = {}


def sec_json(url, nome):
    f = FILINGS / nome
    if f.exists():
        return json.loads(f.read_text())
    d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=ua()), timeout=60).read())
    f.write_text(json.dumps(d))
    return d


def search_filings(ticker, form="10-K", fiscal_year=None):
    tk = sec_json("https://www.sec.gov/files/company_tickers.json", "company_tickers.json")
    cik = next((v["cik_str"] for v in tk.values() if v["ticker"].upper() == str(ticker).upper()), None)
    if cik is None:
        return {"error": f"unknown ticker {ticker}"}
    rec = sec_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"sub_{cik}.json")["filings"]["recent"]
    out = []
    for i, f in enumerate(rec["form"]):
        if f != form or (fiscal_year and not rec["reportDate"][i].startswith(str(fiscal_year))):
            continue
        acc = rec["accessionNumber"][i]
        out.append({"key": acc, "form": f, "filed": rec["filingDate"][i], "period": rec["reportDate"][i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{rec['primaryDocument'][i]}"})
        URLS[acc] = out[-1]["url"]
    return out[:5]


def texto_filing(key):
    f = FILINGS / f"{key}.txt"
    if f.exists():
        return f.read_text(encoding="utf-8")
    raw = urllib.request.urlopen(urllib.request.Request(URLS[key], headers=ua()), timeout=120).read().decode("utf-8", "ignore")
    raw = re.sub(r"(?is)<ix:header>.*?</ix:header>|<script.*?</script>|<style.*?</style>", " ", raw)
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).replace("\xa0", " ")
    f.write_text(t, encoding="utf-8")
    return t


# ------------------------------------------------------------------ calculator (arithmetic only)
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
       ast.UAdd: op.pos}


def calc(e):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError("arithmetic only")
    try:
        return str(round(ev(ast.parse(str(e).replace(",", "").replace("%", "").replace("$", ""), mode="eval").body), 6))
    except Exception as x:
        return f"error: {x}"


# ------------------------------------------------------------------ Agentic RAG
TOOLS = [
    {"type": "function", "function": {"name": "search_filings",
     "description": "List a company's SEC filings on EDGAR by form and fiscal year. Returns filing keys to read.",
     "parameters": {"type": "object", "properties": {"ticker": {"type": "string"},
                    "form": {"type": "string", "description": "10-K, 10-Q, 8-K ..."}, "fiscal_year": {"type": "integer"}},
                    "required": ["ticker", "form"]}}},
    {"type": "function", "function": {"name": "read_filing",
     "description": "Ask one focused question about one filing. An analyst reads the whole filing and returns the relevant figures and quotes.",
     "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "question": {"type": "string"}},
                    "required": ["key", "question"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate arithmetic, e.g. 195201/416161*100",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
SIS_AGENTE = ("You are a financial research agent working from SEC EDGAR filings. Plan the company, form and periods you need, "
              "search the filings, read them with focused questions, use the calculator for any arithmetic, and check the "
              "evidence before answering. Use only figures you read in filings. When the evidence answers the question, "
              "reply with the final answer only: one or two sentences with the figures and units.")
SIS_LEITOR = ("You read an SEC filing for an analyst. Answer the question only from the filing, quoting the exact figures "
              "(with units and periods) and the sentences they come from. Be brief.")
SIS_RESPOSTA = ("You answer questions about SEC filings using only the passages given. Reply in one or two sentences with "
                "the figures and units; compute any change or ratio the question asks for. If the passages do not contain "
                "the answer, say so.")


def agentic(qi, q):
    t0 = time.time()
    msgs = [{"role": "system", "content": SIS_AGENTE}, {"role": "user", "content": f"Question about Apple Inc. (ticker AAPL): {q}"}]
    rec = dict(qi=qi, metodo="agentic_rag", question=q, llm_calls=0, llm_tokens_in=0, llm_tokens_out=0, ds_cost=0.0,
               ds_cold=0.0, jev_calls=0, jev_cost=0.0, reads=0, trace=[])

    def conta(u, a, c):
        rec["llm_calls"] += 1
        rec["llm_tokens_in"] += u.get("prompt_tokens", 0)
        rec["llm_tokens_out"] += u.get("completion_tokens", 0)
        rec["ds_cost"] += a
        rec["ds_cold"] += c
    resposta = "(no final answer in 8 turns)"
    for _ in range(8):
        m, u, a, c = deepseek(msgs, qi, "agentic_rag", "agent_turn", tools=TOOLS)
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
                res = search_filings(args.get("ticker", ""), args.get("form", "10-K"), args.get("fiscal_year"))
                out = json.dumps([{k: x[k] for k in ("key", "form", "filed", "period")} for x in res]
                                 if isinstance(res, list) else res)
                rec["trace"].append(f"search {args}")
            elif nome == "read_filing":
                key = args.get("key", "")
                if rec["reads"] >= 3:
                    out = "read limit reached; answer with the evidence you have"
                elif key not in URLS and not (FILINGS / f"{key}.txt").exists():
                    out = "unknown key; call search_filings first"
                else:
                    rec["reads"] += 1
                    txt = texto_filing(key)
                    rm, ru, ra, rc = deepseek([{"role": "system", "content": SIS_LEITOR},
                                               {"role": "user", "content": f"FILING {key}:\n{txt}\n\nQUESTION: {args.get('question', q)}"}],
                                              qi, "agentic_rag", "read_filing")
                    conta(ru, ra, rc)
                    out = (rm.get("content") or "").strip() or "(the reader returned nothing)"
                    rec["trace"].append(f"read {key}: {args.get('question')}")
            elif nome == "calculator":
                out = calc(args.get("expression", ""))
                rec["trace"].append(f"calc {args.get('expression')} = {out}")
            else:
                out = "unknown tool"
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    rec.update(answer=resposta, correct=nota(qi, resposta), seconds=time.time() - t0)
    return rec


# ------------------------------------------------------------------ Jev RAG
def carregar_e5():
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    m = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(dev).eval()

    def emb(texts):
        vs = []
        for i in range(0, len(texts), 16):
            b = tk(texts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = m(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1)
            vs.append(torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=-1))
        return torch.cat(vs)
    pv = emb(["passage: " + p["text"] for p in rb.P])
    return lambda q: rb.rank_of((emb(["query: " + q]) @ pv.T)[0].tolist())


def jev_rag(qi, q, e5):
    t0 = time.time()
    fil = search_filings("AAPL", "10-K")[0]                  # code: company and form are lookups, latest 10-K
    trace = [f"code: {fil['form']} period {fil['period']}"]
    assert fil["key"] == SEED, fil                           # passages rb.P come from this filing's text
    ordem = rb.rrf(rb.rank_of(rb.bm25(q)), e5(q))
    jev_calls, jev_cost = 0, 0.0
    for rodada in range(2):                                  # answer not there: the next 30
        tj = time.time()
        ids, existe, u = rb.jev_choice(q, ordem[rodada * 30:(rodada + 1) * 30])
        c = float(u.get("cost", 0) or 0)
        jev_calls += 1
        jev_cost += c
        log_call(t=time.time(), qi=qi, metodo="jev_rag", tipo="jev_choice", seg=round(time.time() - tj, 2), usage=u, jev_cost=c)
        trace.append(f"jev: top {ids[:2]}, answer there p={existe:.2f}")
        if existe >= 0.5:
            break
    top = ids[:2]
    t_ret = time.time() - t0
    ps = "\n\n".join(f"[{i}] {next(p['text'] for p in rb.P if p['id'] == i)}" for i in top)
    msgs = [{"role": "system", "content": SIS_RESPOSTA},
            {"role": "user", "content": f"Passages from Apple's 10-K for fiscal 2025 ($ in millions unless stated):\n\n{ps}\n\nQuestion: {q}"}]
    base = dict(qi=qi, question=q, passages=top, gold_in_top2=any(i in rb.GOLD[qi - 1] for i in top), answer_there=existe,
                jev_calls=jev_calls, jev_cost=jev_cost, reads=0, trace=trace)
    out = []
    t1 = time.time()
    m, u, a, c = deepseek(msgs, qi, "jev_rag_deepseek", "answer", max_tokens=3000)
    ans = (m.get("content") or "").strip()
    out.append(dict(base, metodo="jev_rag_deepseek", answer=ans, correct=nota(qi, ans), llm_calls=1,
                    llm_tokens_in=u.get("prompt_tokens", 0), llm_tokens_out=u.get("completion_tokens", 0),
                    ds_cost=a, ds_cold=c, seconds=t_ret + time.time() - t1))
    t1 = time.time()
    ans, tin, tout = qwen_local(msgs)
    log_call(t=time.time(), qi=qi, metodo="jev_rag_qwen_local", tipo="answer", seg=round(time.time() - t1, 2),
             usage={"prompt_tokens": tin, "completion_tokens": tout})
    out.append(dict(base, metodo="jev_rag_qwen_local", answer=ans, correct=nota(qi, ans), llm_calls=1, llm_tokens_in=tin,
                    llm_tokens_out=tout, ds_cost=0.0, ds_cold=0.0, seconds=t_ret + time.time() - t1))
    return out


# ------------------------------------------------------------------ run and summary
def resumo():
    if not RESULTS.exists():
        return
    rs = [json.loads(l) for l in RESULTS.open(encoding="utf-8")]
    for r in rs:
        if (r["qi"], r["metodo"]) in MANUAL:
            r["correct"] = False
    comuns = set.intersection(*[{r["qi"] for r in rs if r["metodo"] == m} for m in {r["metodo"] for r in rs}])
    print(f"\n{len(comuns)} questions answered by every method\n")
    print(f"{'method':22s} right   LLM calls  tokens read  DeepSeek $ as run  $/question no cache  Jev $   seconds")
    tab = {}
    for m in ("agentic_rag", "jev_rag_deepseek", "jev_rag_qwen_local"):
        x = [r for r in rs if r["metodo"] == m and r["qi"] in comuns]
        if not x:
            continue
        n = len(x)
        tab[m] = dict(n=n, right=sum(r["correct"] for r in x), llm_calls=sum(r["llm_calls"] for r in x) / n,
                      tokens_read=sum(r["llm_tokens_in"] for r in x) / n, ds_cost=sum(r["ds_cost"] for r in x),
                      cold_per_q=sum(r["ds_cold"] for r in x) / n, jev_cost=sum(r["jev_cost"] for r in x),
                      seconds=sum(r["seconds"] for r in x) / n)
        t = tab[m]
        print(f"{m:22s} {t['right']:>2}/{n:<3} {t['llm_calls']:9.1f}  {t['tokens_read']:11,.0f}  {t['ds_cost']:17.4f}  "
              f"{t['cold_per_q']:19.5f}  {t['jev_cost']:.4f}  {t['seconds']:7.1f}")
    print(f"\nDeepSeek spent in this log: ${gasto():.4f} (cap ${CAP})")
    (HERE / "rag_e2e_summary.json").write_text(json.dumps(tab, indent=1))


def main():
    if "--summary" in sys.argv:
        return resumo()
    only = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
    feitos = {(r["qi"], r["metodo"]) for r in map(json.loads, RESULTS.open(encoding="utf-8"))} if RESULTS.exists() else set()
    print(f"DeepSeek balance ${saldo():.2f}; spent in this log so far ${gasto():.4f}")
    e5 = carregar_e5()
    try:
        for qi, (q, _) in enumerate(rb.QS, 1):
            if only and qi != only:
                continue
            print(f"q{qi}: {q}", flush=True)
            if not {(qi, "jev_rag_deepseek"), (qi, "jev_rag_qwen_local")} <= feitos:
                for rec in jev_rag(qi, q, e5):
                    log_result(rec)
            if (qi, "agentic_rag") not in feitos:
                log_result(agentic(qi, q))
    except SemCredito as e:
        print("STOPPED:", e)
    resumo()
    print(f"DeepSeek balance now ${saldo():.2f} (the balance page can lag a few minutes)")


if __name__ == "__main__":
    main()
