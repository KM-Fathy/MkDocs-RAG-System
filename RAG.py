import os
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")

genai.configure(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="mkdocs_db/")

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=api_key,
    model_name="models/text-embedding-004"
)

collection = chroma_client.get_collection(
    name="MkDocsGuide",
    embedding_function=google_ef
)
print(f"✅ Connected to Vector DB. Document count: {collection.count()}")

def retrieve_context(query: str, n_results: int = 6):
    """
    Searches the vector DB for chunks related to the query.
    Retrieves 6 chunks to ensure sufficient context.
    """

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if not results["documents"]:
        return [], []
        
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return documents, metadatas
    

def generate_answer(query: str):
    """
    Full RAG pipeline: Retrieve Context -> Construct Prompt -> Generate Answer
    """
    print(f"\n🔍 Searching documentation for: '{query}'...")
    
    docs, metas = retrieve_context(query)
    
    if not docs:
        return "I cannot answer this based on the provided documentation."

    context_parts = []
    for doc, meta in zip(docs, metas):
        source_file = meta.get('source', 'Unknown File')
        context_parts.append(f"--- Source: {source_file} ---\n{doc}")
    
    full_context = "\n\n".join(context_parts)

    prompt = f"""You are a specialized technical support assistant for MkDocs.
Your task is to answer the user's question using ONLY the provided context snippets below.

STRICT RULES:
1. Use ONLY the information present in the 'Context' section.
2. Do NOT use any prior knowledge, outside information, or training data.
3. If the answer is not explicitly found in the Context, you MUST reply exactly with: "I cannot answer this question based on the provided documentation."
4. Do not hallucinate or make up features that are not mentioned.
5. If providing code/config examples, use the exact format from the context.

Context:
{full_context}

User Question:
{query}
"""

    model = genai.GenerativeModel('gemini-2.0-flash') 
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    print("\n🤖 MkDocs AI Assistant (Stable) is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\n❓ Ask a question about MkDocs: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        answer = generate_answer(user_input)
        print("\n💡 Answer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)