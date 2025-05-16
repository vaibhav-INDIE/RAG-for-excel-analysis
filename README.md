# 🤖 Gemini-RAG Chatbot with Excel Data and Vector Search

This project creates an AI chatbot that enables conversational queries over Excel files using **Google Gemini models** and **FAISS vector search**.

It uses:
- `models/text-embedding-004` from Gemini for vector embeddings
- `gemini-1.5-flash` for LLM analysis and response generation

---

## ✨ Features

- 📥 Upload and parse Excel (.xlsx) files
- 🔡 Flatten Excel content into textual chunks
- 🧠 Generate semantic vector embeddings using Gemini's `text-embedding-004`
- 💾 Store vectors in a FAISS index
- 🔍 Retrieve relevant content with similarity search
- 💬 Analyze and answer queries with `gemini-1.5-flash`

---

## 🧰 Tech Stack

- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://python.langchain.com/)
- [Pandas](https://pandas.pydata.org/)
- Python 3.9+

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/vectordb-llm-chats.git
cd vectordb-llm-chats
