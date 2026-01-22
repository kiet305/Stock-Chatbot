from google.adk.agents import LlmAgent
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from sqlalchemy import create_engine

load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PROMPT = """
    You are a Retrieval-Augmented Generation chatbot to answer question related to Stocks and Finances
    You have access to a tool that retrieves context from a blog post,
    Use the tool to help answer user queries.
"""

def rag_search(query):
    """Retrieve Vector DB and and return relevant chunks"""
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="my_documents",
        url="http://localhost:6333",
    )
    results = qdrant.similarity_search(
        query, k=3
    )
    return results

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='rag_chatbot',
    description='A helpful assistant to help with performing Vector DB (Qrant) searches a',
    instruction=PROMPT,
    tools=[rag_search],
)
