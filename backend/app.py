import json
from flask import Flask, render_template, request, jsonify
import requests
from rag import retrieve_context

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "medllama2"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    # Get RAG context
    context = retrieve_context(user_input)

    # Prompt for Ollama
    prompt = f"""
    You are a helpful and empathetic medical assistant chatbot. 
    Use the retrieved medical context below and the user's input to generate your answer.

    Your responsibilities:
    1. Provide a simple analysis of what the user's symptoms might generally indicate (avoid complex jargon).
    2. Suggest preventive measures and lifestyle tips.
    3. Offer safe and practical medical advice, including when the user should consult a doctor.
    4. If symptoms seem severe, urgent, or life-threatening, clearly recommend seeking immediate medical help.

    Format your response in this structure:
    **Possible Causes:** 
    - List 2-3 common or likely explanations.

    **Preventive Measures:** 
    - Provide practical steps or lifestyle changes.

    **Medical Advice:** 
    - Suggest safe next steps, home care, or when to seek professional care.

    **Disclaimer:** 
    ⚠️ This is general health information, not a substitute for professional medical advice. 
    Always consult a qualified healthcare professional for diagnosis and treatment.

    Context: {context}
    User: {user_input}
    """

    # Call Ollama with streaming
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt
    }, stream=True)

    bot_reply = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    bot_reply += data["response"]
            except:
                pass

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
