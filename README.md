# Semantic Search with OpenAI and FAISS

## **Overview**

This project demonstrates the integration of a vector database with a Large Language Model (LLM) to create an efficient semantic search system. By leveraging OpenAI's `text-embedding-ada-002` model for generating text embeddings and FAISS (Facebook AI Similarity Search) for managing high-dimensional vectors, the system enables meaningful and relevant search results based on the semantic similarity of textual content.

## **Features**

- **Embeddings Generation:** Utilizes OpenAI's API to convert textual data into high-dimensional embeddings that capture the semantic essence of the text.
- **Vector Database:** Implements FAISS for efficient storage, indexing, and retrieval of embeddings, enabling rapid similarity searches.
- **Data Collection:** Gathers and processes textual content from multiple open-access sources related to Quantum Computing.
- **Semantic Search:** Provides functionality to perform similarity searches based on user queries, retrieving the most relevant documents.
- **Algorithm Comparison:** Experiments with different similarity measures (cosine similarity and Euclidean distance) to evaluate and optimize search performance.

## **Project Workflow**

### **1. Data Collection and Preparation**

- **Data Sources:** The project collects textual content from seven open-access URLs related to Quantum Computing, ensuring a diverse and comprehensive dataset.
- **Content Extraction:** Utilizes web scraping techniques with `requests` and `BeautifulSoup` to fetch and parse HTML content from each URL. Extracts meaningful text from paragraph tags, filtering out irrelevant or empty content to maintain data quality.

### **2. Embeddings Generation**

- **Model Selection:** Employs OpenAI's `text-embedding-ada-002` model to generate 1536-dimensional embeddings for each text document. This model is chosen for its ability to produce high-quality, semantically rich embeddings.
- **Batch Processing:** Processes texts in batches to optimize API usage, manage rate limits, and enhance efficiency. Incorporates error handling to address potential issues such as empty inputs or API rate constraints.
- **Embedding Storage:** Converts the generated embeddings into NumPy arrays and normalizes them to unit length. This normalization is essential for enabling cosine similarity-based searches using FAISS.

### **3. Database Setup with FAISS**

- **Index Initialization:** Initializes a FAISS index tailored for cosine similarity by using an inner product search after normalization. This setup ensures that the similarity search accurately reflects the semantic closeness between documents.
- **Embedding Ingestion:** Adds the normalized embeddings to the FAISS index, facilitating efficient and scalable similarity searches.
- **Mapping Management:** Maintains a mapping between FAISS index IDs and their corresponding documents, including URLs and content snippets. This mapping is crucial for retrieving and displaying relevant documents based on search results.

### **4. Semantic Search Implementation**

- **Query Embedding:** Converts user queries into embeddings using the same OpenAI model, ensuring consistency between document and query representations.
- **Similarity Search:** Performs similarity searches within the FAISS index to identify the top `k` most similar embeddings to the query embedding.
- **Result Retrieval:** Fetches and displays the corresponding documents for the top matches, providing users with relevant and contextually appropriate information based on their queries.

### **5. Testing and Evaluation**

- **Automated Testing:** Conducts a series of tests using diverse queries related to Quantum Computing to assess the effectiveness and relevance of the search results.
- **Performance Metrics:** Evaluates the system based on the relevance of retrieved documents, the accuracy of similarity measures, and the efficiency of search operations.
- **Feedback Integration:** Analyzes testing outcomes to identify areas for improvement, such as enhancing data quality, optimizing embedding generation, or refining search algorithms.

### **6. Bonus: Experimentation with Similarity Measures**

- **Cosine Similarity vs. Euclidean Distance:** Explores different similarity measures to determine their impact on search performance. Compares cosine similarity, which measures the cosine of the angle between vectors, with Euclidean distance, which measures the straight-line distance between vectors in high-dimensional space.
- **Performance Analysis:** Assesses which similarity measure provides more accurate and relevant search results based on the dataset and application requirements, offering insights into optimizing the search system.

## **Conclusion**

This project successfully integrates OpenAI's `text-embedding-ada-002` model with FAISS to create a robust semantic search system. By following a structured workflow encompassing data collection, embedding generation, database setup, and search implementation, the system effectively retrieves relevant documents based on semantic similarity. Additionally, experimentation with different similarity measures provides valuable insights for optimizing search performance, making the system adaptable to various application needs.

## **Additional Notes**

- **API Rate Management:** The system incorporates batching and delay mechanisms to manage OpenAI API rate limits effectively, ensuring smooth and uninterrupted operation.
- **Data Quality Assurance:** Emphasizes the importance of collecting high-quality, relevant data and implementing robust content extraction methods to maintain the integrity and usefulness of the embeddings.
- **Scalability Considerations:** FAISS is chosen for its scalability and efficiency in handling large-scale similarity searches, making the system capable of managing extensive datasets with high-dimensional vectors.
- **Future Enhancements:** Potential improvements include expanding the dataset, experimenting with more advanced FAISS indices, implementing caching mechanisms for frequent queries, and adding features like filtering, ranking, or grouping of search results based on relevance scores to further enhance the user experience.

---
