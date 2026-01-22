from google.adk.agents import Agent
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from google.adk.tools import google_search
import numpy as np
load_dotenv()
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PROMPT = """
    You are a Retrieval-Augmented Generation chatbot to answer question related to Stocks and Finances
    You have access to a tool that retrieves context from a Vector Database (Qdrant). Please follow the instructions below:
    Step 1: Using 'rag_search' tool to retrieve relevant chunks from the Vector DB based on the user's query.
    Step 2: Evaluate the accuracy of the chunks, ensuring that they answer the question.
    Step 3: If the retrieved chunks are not relevant, using tool 'google_search' to search the web for accurate information to answer the user's question.
    Step 4: If the retrieved chunks are relevant, use them to generate a concise and accurate answer to the user's question.
"""

def rag_search(query):
    """Retrieve Vector DB and and return relevant chunks"""
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="simple_stock",
        url="http://qdrant:6333",
    )
   
   
    results = qdrant.similarity_search_with_score(query, k=58)

    # Step 2: thống kê để tính threshold
    scores = [score for (_, score) in results]

    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    threshold = mean_score + std_dev

    print(f"Mean score: {mean_score}, Std: {std_dev}")

    hybrid_results = []

    for doc, dense_score in results:
        
        hybrid_score = dense_score

        # Chỉ giữ nếu qua ngưỡng thống kê
        if dense_score >= threshold:
            hybrid_results.append((doc, hybrid_score))

    # Step 4: Sắp xếp lại theo hybrid score
    hybrid_results.sort(key=lambda x: x[1], reverse=True)

    return hybrid_results[:5]

root_agent = Agent(
    model='gemini-2.5-flash',
    name='rag_chatbot',
    description='A helpful assistant to help with performing Vector DB (Qrant) searches and web searches to answer questions related to Stocks and Finances.',
    instruction=PROMPT,
    tools=[rag_search],
)
