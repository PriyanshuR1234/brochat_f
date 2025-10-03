import os
import threading
import time
import requests
from flask import Flask, request, jsonify # Removed 'Response'
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document 

# --- RAG Components Initialization ---
# These will be initialized once when the server starts
rag_retriever = None
EMBEDDING_MODEL = "models/gemini-embedding-001" # Highly cost-efficient embedding model
VECTOR_STORE_PATH = "faiss_index" # Local directory to save the index

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for frontend

# Load reference text from file
def load_text_file(path):
    if not os.path.exists(path):
        # IMPORTANT: Ensure your 'data/yourfile.txt' exists when deploying
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --- RAG Indexing Phase (Only run once at startup) ---
def initialize_rag_components(doc_text):
    """Chunks, embeds, and loads the RAG components."""
    global rag_retriever
    
    # 1. Check if the FAISS index already exists on disk
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"[RAG] Loading existing index from {VECTOR_STORE_PATH}...")
        embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        # IMPORTANT: allow_dangerous_deserialization=True is required for loading
        # This fixes the "faiss" dependency issue you had earlier.
        try:
            vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings_model, 
                allow_dangerous_deserialization=True
            )
            rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            print("[RAG] Index loaded successfully.")
        except Exception as e:
             # Handle case where index exists but fails to load (e.g., missing faiss dependency, bad file)
            print(f"[RAG ERROR] Failed to load index: {e}. Rebuilding index.")
            _build_new_index(doc_text, embeddings_model)
        return

    # 2. Indexing Phase (Only if index doesn't exist)
    _build_new_index(doc_text, GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL))

def _build_new_index(doc_text, embeddings_model):
    """Internal function to build a new FAISS index."""
    global rag_retriever
    print("[RAG] Index not found. Creating a new one...")
    
    document = Document(page_content=doc_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  
        chunk_overlap=200, 
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = text_splitter.split_documents([document])
    print(f"[RAG] Document split into {len(chunks)} chunks.")

    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"[RAG] New index created and saved to {VECTOR_STORE_PATH}.")

    rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# --- Server Startup ---
try:
    doc_text = load_text_file("data/yourfile.txt")
    initialize_rag_components(doc_text) # Initialize RAG on startup
except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}. Please ensure the data file exists.")
    exit(1)

# Initialize Gemini model once (for final generation)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-lite-001",
    temperature=0.2
)

# ------------------- Health Check -------------------
target_server = "https://monitor-server-8kgp.onrender.com/health"

def check_health_loop():
    while True:
        try:
            res = requests.get(target_server, timeout=5)
            if res.status_code == 200:
                print(f"[✅ Healthy] {target_server} at {time.strftime('%H:%M:%S')}")
            else:
                print(f"[⚠️ Issue] {target_server} returned {res.status_code} at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"[❌ Down] {target_server} at {time.strftime('%H:%M:%S')} - {e}")
        # Changed back to 3 seconds for better monitoring feedback
        time.sleep(300) 

# Run health check in a separate background thread
threading.Thread(target=check_health_loop, daemon=True).start()

# ------------------- Routes -------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Server is healthy"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Please provide a 'query' field in JSON"}), 400

        query = data["query"]

        # 1. Retrieval (FAISS)
        if rag_retriever is None:
            return jsonify({"error": "RAG component failed to initialize."}), 500
            
        retrieved_docs = rag_retriever.invoke(query)
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        # 2. Generation Prompt (Using the retrieved context)
        prompt = f"""
You are ChatBro, the official assistant for the BrokeBro website (https://www.brokebro.in). 
You help users with questions **only related to BrokeBro**, using the following notes as your primary reference:
Notes (Context):
{context}

Original Document Text (For general context/rules):
{doc_text}

Rules for responding:
1. Answer **only based on the notes** provided. If the answer is in the notes, respond clearly and naturally like a human.  
2. If the answer is **not in the notes**, you may refer to the official website (https://www.brokebro.in) for additional information.  
3. If the information is still unavailable, respond politely: "Sorry, I don’t have that information right now."  
4. Keep your responses concise, friendly, and human-like.  
5. Do not mention AI, ChatGPT, or your system capabilities in the answers.  
6. IIf someone ask for offers so give all types of offers heading and ask user to which you are looking for? or in which the user is interested for details?
7.don't give whole details at once only give headings first then if user ask for specific one then give details for that specific one.
8. Send response heading start with ## and subheading inside double stars **text** and content is simple in the response.

Now answer the user question: {query}
"""
        
        # 3. Synchronous Generation (llm.invoke())
        # This will wait for the entire response to be generated before sending it back.
        response = llm.invoke(prompt)
        
        # 4. Return the response in JSON format
        return jsonify({"answer": response.content}) # <--- JSON response format

    except Exception as e:
        # Catch all errors to prevent server crash and provide feedback
        print(f"An error occurred in /chat: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))