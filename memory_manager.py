from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
import os

persist_directory = "dora_memory"
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def save_memory(text_to_remember: str):
    """
    Saves a piece of text to the vector database.
    """
    try:
        db.add_texts([text_to_remember])
        print(f"🧠 Memory saved: '{text_to_remember}'")
        return "OK, I've remembered that."
    except Exception as e:
        print(f"Error saving memory: {e}")
        return "Sorry, I had trouble remembering that."

def recall_memories(query: str, num_results: int = 2) -> list[str]:
    """
    Searches the vector database for memories relevant to the user's query.
    """
    try:
        print(f"🧠 Searching memory for: '{query}'")
        results = db.similarity_search(query, k=num_results)
        retrieved_texts = [doc.page_content for doc in results]
        return retrieved_texts
    except Exception as e:
        print(f"Error recalling memories: {e}")
        return []