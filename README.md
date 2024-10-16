# Retrieval-Augmented Generation (RAG) Model Using Langchain

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** model using **Langchain** and **OpenAI**. The RAG model combines a **retrieval component** (to retrieve relevant information from a corpus) and a **generation component** (to generate coherent text based on the retrieved information). The model utilizes FAISS for vector-based retrieval and GPT-4o-mini for text generation.

## Project Structure

The project includes the following key steps:

1. **Corpus Preparation**
2. **Vectorization**
3. **Retrieval**
4. **Generation**
5. **Combination of Retrieval and Generation (RAG)**
6. **Evaluation**
7. **Streamlit run**

---

## 1. Corpus Preparation

The corpus is fetched from a domain-specific source (in this case, a blog about quantum computing), cleaned, and split into manageable chunks using Langchain's `RecursiveCharacterTextSplitter`.

### Features:
- Clean text extraction from a webpage.
- Splitting of text into chunks with a 100-character overlap for context continuity.

---

## 2. Vectorization

We used **sentence-transformers/all-mpnet-base-v2**, a pre-trained model from HuggingFace, to generate embeddings for the corpus chunks. These embeddings are stored in a **FAISS vector store** for efficient similarity-based document retrieval.

### Features:
- Embedding generation using a pre-trained model from HuggingFace.
- FAISS vector store for fast and efficient retrieval.

---

## 3. Retrieval Component

A retriever was created using the FAISS vector store. This component enables the retrieval of relevant documents based on a query. FAISS performs vector-based similarity searches to find the most relevant corpus chunks.

### Features:
- FAISS-based retrieval for efficient document search.
- Capable of retrieving the most relevant documents for any given query.

---

## 4. Generation Component

We used **OpenAI's GPT-4o-mini** to generate coherent answers based on the retrieved documents. This component processes the retrieved chunks and formulates responses to user queries.

### Features:
- Generation of responses based on context from retrieved documents.
- Utilizes OpenAI's GPT-4o-mini model to generate coherent answers.

---

## 5. Combine Retrieval and Generation (RAG)

We combined the retrieval and generation components into a RAG model using Langchain’s `RetrievalQA`. This enables the model to retrieve relevant documents and generate responses in an integrated pipeline.

### Features:
- Combines document retrieval and response generation into a unified model.
- Processes queries and generates responses based on relevant retrieved documents.

---

## 6. Evaluation

We used the **ROUGE** metric to evaluate the generated responses by comparing them to a reference answer. The ROUGE score helps measure how similar the generated response is to the reference text.

### Features:
- Evaluation using ROUGE to compare generated responses to a reference answer.
- ROUGE measures how similar the generated text is to the target reference.

---

## 7. Streamlit run

![Streamlit run](https://github.com/user-attachments/assets/10840766-702e-4b46-92f7-a0ea91fae73d)

---

## Summary

This project demonstrates the following:

- **Corpus Preparation**: Cleaned, preprocessed, and split text into chunks using `TextSplitter`.
- **Embedding Generation**: Generated embeddings using the `all-mpnet-base-v2` model and stored them in a FAISS vector store.
- **Retrieval Component**: Implemented FAISS-based document retrieval.
- **Generation Component**: Integrated OpenAI’s GPT-4o-mini model for answer generation.
- **RAG Model**: Combined retrieval and generation into a single pipeline using Langchain’s `RetrievalQA`.
- **Evaluation**: Used ROUGE for performance evaluation and comparison against reference answers.
- **Streamlit**: Ran the .py script in a terminal to trigger the streamlit run.




