import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    # Load your text file (inside data/ folder)
    doc_text = load_text_file("data/yourfile.txt")

    # Initialize Gemini chat model
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite-001", temperature=0.2)

    print("Chatbot ready! Type 'quit' to exit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["quit", "exit", "bye"]:
            break

        # Send one request: your doc + user query
        prompt = f"""
You are a helpful assistant. 
Here is the reference text:\n{doc_text}\n
Now answer the user question: {query}
"""
        response = llm.invoke(prompt)
        print("Bot:", response.content)

if __name__ == "__main__":
    main()
