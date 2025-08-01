# RAG-Xpress — FAISS + LangChain + Cross-Encoder Reranker

_A hyper-efficient, production-grade Retrieval-Augmented Generation stack_

---

> **RAG-Xpress** turbo-charges your LLM with _zero-hallucination_ knowledge retrieval.  
> It blends **heading-aware chunking**, a **hybrid FAISS vector index**, and a **state-of-the-art cross-encoder reranker** to surface bullet-proof evidence in milliseconds.

---

## Key Features

| Category       | Highlights                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| **Embeddings** | `BAAI/bge-base-en-v1.5`                                                                               |
| **Reranking**  | `BAAI/bge-reranker-base`                                                                              |
| **Chunker**    | Regex-aware _RecursiveCharacterTextSplitter_ (800 tokens / 100 overlap) keeps **Excl-clauses** intact |
| **Indexing**   | FAISS + L2-normalized vectors = instant cosine similarity                                             |

---

## Tech Stack

| Layer                | Tech                                    |
| -------------------- | --------------------------------------- |
| **Language Runtime** | Python 3.11                             |
| **Framework**        | **LangChain 0.2 Runnable graph**        |
| **Vector DB**        | **FAISS**                               |
| **Model Hub**        | Sentence-Transformers & 🤗 Transformers |
| **LLM**              | Gemini 2.5 Flash                        |

---
