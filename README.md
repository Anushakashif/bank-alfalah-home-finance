# 🏠 Bank Alfalah Home Finance AI Assistant

A RAG-based (Retrieval-Augmented Generation) AI chatbot that answers customer questions about Bank Alfalah's Home Finance products — instantly, accurately, and strictly from official policy data.

> **Live Demo:** [your-app-url.streamlit.app]([https://share.streamlit.io](https://anushakashif-bank-alfalah-home-finance-app-b3mnhf.streamlit.app/))

---

## 📌 Problem Statement

Customers looking for home finance information are forced to read lengthy policy documents full of legal language just to answer simple questions like:

- *"What is the markup rate for a salaried person?"*
- *"What documents do I need to apply?"*
- *"Do I qualify with a salary of PKR 80,000?"*

This wastes customer time and overloads bank support agents with routine queries that could be resolved instantly.

---

## ✅ Solution

An AI assistant that reads Bank Alfalah's official Home Finance policy and answers customer questions in seconds — in plain English or Urdu — without any human agent involvement.

---

## ⚙️ How RAG Works

Unlike a regular chatbot that relies on general AI knowledge, this system uses **Retrieval-Augmented Generation (RAG)** — meaning it only answers from the actual policy document. Nothing is made up.

```
Customer Question
      ↓
Retrieve relevant policy sections  ←── Knowledge Base (Bank Alfalah Policy)
      ↓
Send only those sections to AI
      ↓
AI generates accurate, grounded answer
      ↓
Customer gets instant response
```

**Three core steps:**

1. **Chunk** — Policy document is split into sections by topic (rates, eligibility, documents, etc.)
2. **Retrieve** — Customer question is matched against all sections using cosine similarity to find the most relevant ones
3. **Generate** — Only the relevant sections are sent to the LLM, which produces a precise answer strictly from that context

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| AI Model | LLaMA 3.1 8B |
| AI Inference | Groq API |
| Retrieval | Cosine Similarity (keyword-based vectors) |
| Knowledge Base | Bank Alfalah Official Home Finance Policy |
| Environment | python-dotenv |

## 💡 Features

- 🔍 Retrieves only relevant policy sections per question — no full document stuffing
- 🛡️ Strictly answers from policy — if something is not covered, it says so honestly
- 🌐 Supports both **English and Urdu** queries
- 📄 Shows which **policy sections were used** for every answer (transparency)
- ⚡ Fast responses via Groq inference
- 🔒 API key secured via environment variables — never exposed in code

---

## 🔮 Future Improvements

- Swap keyword vectors with **sentence-transformers** for true semantic embeddings
- Add **PDF upload** so bank staff can update the knowledge base without touching code
- Extend to cover **Car Loans, Personal Loans, Credit Cards** by adding their policy documents
- Add **conversation memory** for multi-turn follow-up questions
- Integrate with bank's **WhatsApp Business API** for wider reach

---

## ⚠️ Disclaimer

This is a portfolio project built for demonstration purposes. All policy data is sourced from Bank Alfalah's publicly available website. This project is not officially affiliated with or endorsed by Bank Alfalah.
