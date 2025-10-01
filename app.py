import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for frontend

# Load reference text from file
def load_text_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

doc_text = load_text_file("data/yourfile.txt")

# Initialize Gemini model once
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-lite-001",
    temperature=0.2
)
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

        prompt = f"""
You are ChatBro, the official assistant for the BrokeBro website (https://www.brokebro.in). 
You help users with questions **only related to BrokeBro**, using the following notes as your primary reference:

{doc_text}

Rules for responding:
1. Answer **only based on the notes** provided. If the answer is in the notes, respond clearly and naturally like a human.  
2. If the answer is **not in the notes**, you may refer to the official website (https://www.brokebro.in) for additional information.  
3. If the information is still unavailable, respond politely: "Sorry, I don’t have that information right now."  
4. Keep your responses concise, friendly, and human-like.  
5. Do not mention AI, ChatGPT, or your system capabilities in the answers.  

Now answer the user question: {query}
"""

        # Call Gemini
        response = llm.invoke(prompt)

        return jsonify({"answer": response.content})

    except Exception as e:
        # Catch all errors to avoid crashing server
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
