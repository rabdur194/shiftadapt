# ShiftAdapt

**Data-Efficient and Reliable Adaptation of Pretrained/Foundation Models Under Distribution Shift**

A Level-A prototype that combines **RAG + LLM** with **distribution shift detection** and **data-efficient knowledge-base adaptation**.

[![CI](https://github.com/YOUR_USERNAME/shiftadapt/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/shiftadapt/actions)

---

## What it does

1. Builds a RAG knowledge base from an **old domain**
2. Detects **distribution shift** when new data arrives (embedding centroid distance)
3. Measures **reliability** (accuracy + confidence) on the new domain **before** adaptation
4. Adapts in a **data-efficient** way by adding only a few new labeled examples to the knowledge base
5. Re-measures reliability **after** adaptation
6. Uses an **LLM** (or mock LLM) to generate grounded labels and explanations

This directly supports the research theme:
> Data-Efficient and Reliable Adaptation of Pretrained/Foundation Models Under Distribution Shift

---

## Quick start (Docker)

```bash
git clone https://github.com/YOUR_USERNAME/shiftadapt.git
cd shiftadapt

# optional: add OpenAI key
cp .env.example .env

docker compose up --build
```

Open **http://localhost:8501**

---

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export PYTHONPATH=$PWD
export USE_MOCK_LLM=true

streamlit run app.py
```

---

## Project structure

```
shiftadapt/
├── app.py                 # Streamlit demo
├── src/
│   ├── embeddings.py      # Pretrained foundation embeddings
│   ├── shift_detection.py # Distribution shift detection
│   ├── knowledge_base.py  # RAG knowledge base
│   ├── rag_llm.py         # Retrieval + LLM generation
│   ├── adapt.py           # Data-efficient adaptation
│   ├── evaluate.py        # Reliability metrics
│   └── pipeline.py        # End-to-end pipeline
├── data/
│   ├── old_domain/        # Original distribution
│   ├── new_domain/        # Shifted distribution
│   └── adaptation_examples/  # Few new examples for adaptation
├── tests/
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## How it maps to the research proposal

| Research concept | How this project shows it |
|------------------|---------------------------|
| Pretrained / foundation model | `sentence-transformers` embeddings (+ optional LLM) |
| Distribution shift | Centroid distance between old vs new embeddings |
| Data-efficient adaptation | Add only a few labeled examples to the knowledge base |
| Reliability | Accuracy + confidence before vs after adaptation |
| RAG | Retrieval-augmented generation for grounded answers |

---

## Design choices (Level A)

- **No full fine-tuning / LoRA** — keeps the prototype simple and focused
- **Knowledge-base update** as the adaptation method — practical and data-efficient
- **Mock LLM mode** — works without an API key for demos and CI
- **Clear before/after metrics** — easy to explain to a professor or interviewer

### Room to grow (future work)

- Stronger drift detectors (MMD, energy distance)
- Calibration metrics (ECE)
- RAGAS faithfulness evaluation
- Parameter-efficient fine-tuning (LoRA) as a second adaptation path
- Active selection of which examples to add

---

## API key (optional)

By default the app uses a **mock LLM** so it runs offline.

To use a real LLM:

```bash
export OPENAI_API_KEY=sk-...
export USE_MOCK_LLM=false
```

---

## Tests & CI

```bash
export PYTHONPATH=$PWD
pytest tests/ -v
```

GitHub Actions runs tests and a Docker build on every push to `main`.

---

## License

MIT
