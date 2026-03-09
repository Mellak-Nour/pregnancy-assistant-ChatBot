# MamaConnect

<p align="center">
  <img src="logo.png" width="120" alt="MamaConnect Logo"/>
</p>

<p align="center"><b>Pregnancy Health RAG Assistant — Technical Documentation</b></p>

A Retrieval-Augmented Generation (RAG) system acting as a **Personal Assistant for Pregnant Women** — powered by **Gemini API**, **FAISS**, and **SentenceTransformers**.

---

## System Architecture

```
User Question
      ↓
Load User Context (pregnancy week, trimester, allergies, illnesses)
      ↓
Danger Detection ──► Emergency Alert (if severe symptoms detected)
      ↓
Embed Question → FAISS Top-5 Retrieval
      ↓
Build Structured Prompt (persona + profile + retrieved docs + question)
      ↓
Gemini API (gemini-2.5-flash)
      ↓
Structured Response: Answer / Safety Advice / Recommendation
```

---

## 1. Vector Database — FAISS

FAISS (Facebook AI Similarity Search) was chosen because:
- Runs fully locally — no server or cloud setup needed
- Fast exact nearest-neighbor search suited to our dataset size
- Vectors are normalised before indexing, making L2 equivalent to cosine similarity
- Index is saved to disk and reloaded instantly on subsequent runs

---

## 2. Retrieval Strategy — Top-K Similarity Search

1. User question embedded into a **384-dim vector** using `all-MiniLM-L6-v2`
2. Query vector normalised consistently with the index build process
3. **FAISS IndexFlatL2** performs exact search — returns top-5 nearest neighbors
4. The 5 matching chunks are injected into the Gemini prompt as grounded medical context

---

## 3. Prompt Engineering — Context Injection + Persona

| Layer | Purpose |
|-------|---------|
| **Persona** | Safe, evidence-based pregnancy guide — never invents information |
| **User Profile** | Injects week, trimester, allergies, and medical conditions |
| **Retrieved Knowledge** | Top-5 FAISS chunks provide factual grounding |
| **Output Format** | Structured response: Answer / Safety Advice / Recommendation |

Temperature is set to **0.3** for factual answers. Danger keywords (bleeding, severe headache, no fetal movement, etc.) trigger an emergency alert before the LLM call.

---

## 4. Dataset

| File | Rows | Content |
|------|------|---------|
| `foods.csv` | 40+ | Food safety ratings with trimester notes |
| `exercise.csv` | 30 | Exercise safety with trimester modifications |
| `symptoms.csv` | 45 | Symptoms with LOW / MEDIUM / HIGH risk levels |
| `weekly_guidance.csv` | 37 | Week 4–40 baby development, nutrition, exercise tips |

Each row is converted to a natural language sentence, embedded, and stored in FAISS.

---

## Installation & Usage

```bash
pip install google-generativeai sentence-transformers faiss-cpu numpy pandas
export GEMINI_API_KEY="your-key-here"
python pregnancy_rag_assistant.py
```

*MamaConnect — For informational guidance only. Does not replace professional medical advice.*