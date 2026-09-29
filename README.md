<p align="center">
  <a href="https://davidariasfinance.com/research/is-jev-efficient-for-rag/"><img src="assets/banner.jpg" alt="Agentic RAG vs Jev RAG in finance: the agent loops through whole filings, 82,908 tokens read and 46 of 50 right; Jev picks 2 passages, 840 tokens read and 50 of 50 right" width="100%"></a>
</p>

<p align="center">
  <a href="https://davidariasfinance.com/research/is-jev-efficient-for-rag/"><img src="https://img.shields.io/badge/Read_the_article-davidariasfinance.com-0b2545" alt="Read the article"></a>
  <a href="https://typesafe.ai"><img src="https://img.shields.io/badge/Model-Jev_%28TypeSafe%29-e8504a" alt="Jev by TypeSafe"></a>
  <a href="https://openrouter.ai"><img src="https://img.shields.io/badge/Access-OpenRouter-6566f1" alt="OpenRouter"></a>
  <a href="https://github.com/vals-ai/finance-agent-v2"><img src="https://img.shields.io/badge/Baseline-Vals_AI_Finance_Agent-1a1d1b?logo=github" alt="Vals AI Finance Agent"></a>
  <a href="https://www.sec.gov/edgar/search/"><img src="https://img.shields.io/badge/Data-SEC_EDGAR-1f4e79" alt="SEC EDGAR"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2e4bc9" alt="MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <a href="https://x.com/Davidariasfin"><img src="https://img.shields.io/badge/Follow-%40Davidariasfin-000000?logo=x&logoColor=white" alt="Follow on X"></a>
</p>

## What it is

Jev for Finance applies [Jev](https://typesafe.ai) to financial research, starting with RAG over SEC filings.
Retrieval is treated as a decision: code finds the filing, search keeps 30 passages, Jev picks the one that answers,
and an LLM reads two. This repository holds that pipeline (Jev RAG), a benchmark against an agentic RAG harness on 50
analyst questions about four 10-Ks, and every log behind the numbers. Full write-up:
[Is Jev efficient for RAG?](https://davidariasfinance.com/research/is-jev-efficient-for-rag/)

## Results

50 questions on the latest 10-Ks of Apple, Microsoft, Nvidia and Amazon, asked the way an analyst would ("What was
Microsoft's bottom line in its latest fiscal year?", "Why did AWS sales grow?"). Every answer was graded against the
filing and then read by hand.

| Per question, all 50 | Right | LLM calls | Tokens read by the LLM | Seconds | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agentic RAG, DeepSeek Flash | 46 of 50 | 4.7 | 82,908 | 11.1 | $0.0031 cached, $0.0133 uncached |
| Jev RAG, DeepSeek Flash | **50 of 50** | 1 | 840 | 2.0 | $0.0007 |
| Jev RAG, Qwen 3.5 4B (local) | 45 of 50 | 1 | 990 | 12.6 | $0.0006 |

Swapping the LLM on both sides:

| LLM | Agentic RAG | Jev RAG | Agent tokens read per question | Jev RAG tokens read per question |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek Flash | 46 of 50 | 50 of 50 | 82,908 | 840 |
| Qwen 3.7 Flash | 41 of 50 | 49 of 50 | 82,428 | 988 |
| GPT-5 nano | 34 of 50 | 48 of 50 | 64,634 | 815 |

Retrieval alone, correct passage ranked first: BM25 34%, e5 embeddings 52%, BM25 + e5 40%, MiniLM reranker 54%,
bge reranker 36%, Jev 90%. Most agent misses (21 of 29 across the three LLMs) came from opening the previous year's
filing. Total cost of the whole study: about $1.

## Quick start

```bash
git clone https://github.com/davidalmeida90/jev-for-finance
cd jev-for-finance
pip install -r requirements.txt          # torch and transformers are optional, for vector search
cp .env.example .env                     # OPENROUTER_API_KEY and SEC_USER_AGENT
python jev_rag.py MSFT "How much does Microsoft spend on R&D?"
```

```text
MICROSOFT CORP 10-K, period 2026-06-30, filed 2026-07-29 (216 passages)
first stage BM25 + e5 (RRF); Jev: 2 call(s), answer there p=0.97, picked P107 (0.67), P108 (0.33)

In fiscal 2026, Microsoft spent $35,562 million on research and development.

LLM qwen/qwen3.7-flash: 752 tokens read | Jev $0.00092 + LLM $0.00020 | 39.1s
```

Options: `--form 10-Q` for quarterly reports, `--llm <OpenRouter model id>` for another answer writer
(`--llm deepseek-flash` uses DeepSeek's own API with `DEEPSEEK_API_KEY`), `--no-embed` for BM25 alone, `--json` for
the full result. `answer()` in `jev_rag.py` returns the same dictionary for use from Python. Filings are cached in
`cache/` after the first download, and a first run with vector search also downloads `intfloat/multilingual-e5-base`.

## How it works

**Jev RAG** follows TypeSafe's [reranking pattern](https://docs.typesafe.ai/cookbooks/rerank_typesafe):

1. Code looks up the latest filing on EDGAR, since ticker, form and "latest" are lookups.
2. It splits the filing into 1,500-character passages; BM25 and e5 embeddings rank them, fused with reciprocal rank
   fusion.
3. One Jev call (`jev-latest` on OpenRouter's `/api/v1/systemone`) ranks the top 30 with a Choice question and asks
   a Noul question: are these passages likely to contain the answer?
4. If that probability is under 0.5, the next 30 go in.
5. An LLM reads the top two passages and writes the answer.

```python
answers = ask(hybrid[:30])             # one Jev call: Choice "best" + Noul "exists"
if answers["exists"]["noul"] < 0.5:    # answer not in these 30
    answers = ask(hybrid[30:60])       # so try the next 30
p = answers["best"]["probabilities"]
top_two = sorted(p, key=p.get, reverse=True)[:2]  # all the LLM sees
```

**Agentic RAG** follows the [Vals AI Finance Agent](https://github.com/vals-ai/finance-agent-v2) harness. An LLM
agent searches EDGAR, sends the whole filing to a second LLM call with a focused question, uses a calculator and
loops, for at most 8 turns and 3 filing reads. Vals's reading tool can also take a character range; this version
always sends the whole filing. Both sides get the company name, and the agent gets today's date.

## Reproduce the benchmark

Every result is in the logs, so the tables rebuild for free and without keys:

```bash
cd benchmark
python retrieval_bench.py          # retrieval on Apple's 20 questions (reads cached scores)
python rag_e2e.py --summary        # Apple end to end
python rag_multi.py --summary      # all four companies, retrieval, order check, end to end
python rag_models.py --summary     # Qwen 3.7 Flash, GPT-5 nano, Gemini 2.5 Flash-Lite
```

To rerun from scratch, move the `e2e_log*` folders and the cached scores in `bench_cache/*.json` aside; each script
skips work already in its log. A full rerun needs `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY` and `SEC_USER_AGENT`,
[Ollama](https://ollama.com) with `qwen3.5:4b` for the local rows, and about $1 at September 2026 prices. Spend caps
sit at the top of each script.

Automatic grades are regular expressions on the figures in each filing. Where a hand read disagreed, the override and
its reason are in each script's `MANUAL` table (13 overrides in the headline runs, in both directions).

All runs used Python 3.11 on 27 and 28 September 2026. Compared with those runs, only two things changed: keys now
come from environment variables, and Apple's filing text is read from `bench_cache/filings/`.

## Layout

```text
jev_rag.py                  the pipeline as one file (CLI and answer())
benchmark/
  retrieval_bench.py        nine retrieval set-ups on Apple's 10-K
  rag_e2e.py                Apple end to end: Agentic RAG vs Jev RAG
  rag_multi.py              Microsoft, Nvidia, Amazon + order check
  rag_models.py             the same test with three more LLMs
  config.py                 keys and SEC User-Agent from the environment
  bench_cache/              filing texts, EDGAR indexes, cached model scores
  e2e_log*/                 every API call and every graded answer
  *_summary.json            the tables above, as JSON
assets/                     README banner
```

## Limits

- 50 questions on four US large caps with three budget LLMs show a pattern and don't settle it.
- Jev's existence check can sit close to its 0.5 threshold. On Microsoft's R&D question it returned 0.47 in the
  benchmark and 0.52 when the filing was labelled with its EDGAR name, which kept the wrong 30 passages. `jev_rag.py`
  labels filings the way the benchmark did; a higher threshold or an unconditional second round is the safer choice.
- Jev RAG takes the latest filing by design. Questions about earlier years need a date parser.
- `jev_rag.py` calls a 10-K "fiscal YYYY" from the year its period ends, which holds for the four filers tested and
  not for every company.
- Jev runs in TypeSafe's cloud; `jev-latest` resolved to Jev 1.13 during the tests.
- Gemini 2.5 Flash-Lite also ran: 21 of 50 as an agent, 49 of 50 in Jev RAG. It asked the user back or skipped its
  tools on 25 of the 50 agentic questions, so the write-up leaves it out; its rows stay in the logs.

## Data and licenses

- Code: [MIT](LICENSE).
- Filings in `benchmark/bench_cache/filings/` come from SEC EDGAR and are public domain.
- Logs hold outputs of Jev, DeepSeek Flash, Qwen, GPT-5 nano and Gemini, published for reproducibility.
  TypeSafe's terms forbid using Jev output to train a model that imitates it.
- Models fetched at run time: `intfloat/multilingual-e5-base` (MIT), `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (Apache 2.0), `BAAI/bge-reranker-base` (MIT).

Agentic design after [vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2); Jev RAG after
TypeSafe's [reranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe). Not affiliated with TypeSafe.
Nothing here is investment advice.

---

**David Arias, CFA** · [davidariasfinance.com](https://davidariasfinance.com/research/) ·
[X @Davidariasfin](https://x.com/Davidariasfin)
