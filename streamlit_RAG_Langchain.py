# Imports
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from rouge_score import rouge_scorer
from langchain.cache import InMemoryCache
import streamlit as st
import openai
from typing import Optional, List
# Function to fetch and clean text from a URL
def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted HTML tags
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Get all text content
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

# Sample URL for fetching corpus
url = "https://medium.com/@vignesh2659/quantum-computing-abd85aa5da9d"
corpus = fetch_text_from_url(url)

# Split the text into smaller chunks using Langchain's TextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
corpus_chunks = splitter.split_text(corpus)
# Initialize HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create embeddings for the chunks
chunk_embeddings = embedding_model.embed_documents(corpus_chunks)

# Initialize FAISS vector store and add texts with their embeddings
vector_store = FAISS.from_texts(texts=corpus_chunks, embedding=embedding_model)
# Create a retriever using FAISS
retriever = vector_store.as_retriever()

# Example retrieval for a query
query = "What is Superposition?"
retrieved_docs = retriever.get_relevant_documents(query)
    
# Function to generate a response using the GPT-4o-mini model
def generate_response(query, retrieved_docs):
    # Combine the retrieved document content into a single context
    context = " ".join([doc.page_content for doc in retrieved_docs])
    
    # Use OpenAI's API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": query}
        ],
        max_tokens=300,
        temperature=0.7
    )

    # Extract the first choice's content from the response
    return response['choices'][0]['message']['content']
# Custom wrapper for Openai function
class CustomLLM(LLM):
    def _call(self, prompt: str, retrieved_docs: List = [], stop: Optional[List[str]] = None) -> str:
        # Initialize the LLM (gpt-4o-mini) with your OpenAI API key
        openai.api_key = "key"
        return generate_response(prompt, retrieved_docs)

    @property
    def _identifying_params(self) -> dict:
        return {"model": "gpt-4o-mini"}

    @property
    def _llm_type(self) -> str:
        return "custom"

# Initialize the custom LLM and the RAG model
openaillm = CustomLLM()
rag_model = RetrievalQA.from_chain_type(llm=openaillm, retriever=retriever)

# Query using the combined RAG model
query = "Explain the importance of Superposition from the context provided."
response_rag = rag_model.run(query)

# Display the response
print(f"RAG Response: {response_rag}")

# Define a function to calculate ROUGE scores for evaluation
def evaluate_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores

# Example evaluation
reference_answer = "During the spin of a coin, there are 2 possibilities, both head and a tail. That position of a coin having both at the same time is what a qubit is. This state is called Superposition."

# Evaluate the RAG-generated answer using ROUGE
rouge_scores_rag = evaluate_rouge(response_rag, reference_answer)
print("ROUGE Scores RAG:", rouge_scores_rag)
# Initialize the RAG model using Langchain's RetrievalQA with custom LLM
rag_model = RetrievalQA.from_chain_type(llm=openaillm, retriever=retriever)
# Streamlit-based interface for querying the RAG model
st.title("RAG Model Interface")

query = st.text_input("Enter your question:")
if query:
    response_rag = rag_model.run(query)
    st.write(f"Response: {response_rag}")