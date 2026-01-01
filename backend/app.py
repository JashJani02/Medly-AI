import os, json, requests, time
from flask import Flask, render_template, request, jsonify, send_from_directory
from rag import retrieve_context, add_user_file
from gtts import gTTS
from deep_translator import GoogleTranslator as gt

# --- Flask setup ---
app = Flask(__name__)

# --- Folder setup ---
UPLOAD_FOLDER = "uploads"
AUDIO_ROOT = os.path.join(os.path.dirname(__file__), "..", "static", "audio")  # root/static/audio
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_ROOT, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"


# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data["message"]
    model_name = data.get("model", "medllama2")   # chosen in UI
    target_lang = data.get("lang", "en")

    # Retrieve RAG context
    context = retrieve_context(user_input)

    # Prompt for the LLM
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

    # Call Ollama model
    response = requests.post(
        OLLAMA_URL,
        json={"model": model_name, "prompt": prompt, "images": []},
        stream=True
    )

    bot_reply = ""
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                bot_reply += chunk.get("response", "")
            except:
                pass

    # --- Translate reply ---
    try:
        translated_reply = gt(source='auto', target=target_lang).translate(bot_reply)
    except Exception as e:
        print("[Translation error]", e)
        translated_reply = bot_reply

    # --- Convert reply to speech ---
    timestamp = int(time.time())
    audio_filename = f"reply_{timestamp}.mp3"
    audio_path = os.path.join(AUDIO_ROOT, audio_filename)

    try:
        tts = gTTS(text=translated_reply, lang=target_lang)
        tts.save(audio_path)
        print(f"🎧 Saved audio: {audio_path}")
    except Exception as e:
        print(f"[TTS Error] {e}")
        audio_path = None

    return jsonify({
        "reply": bot_reply,
        "audio": f"/static/audio/{audio_filename}" if audio_path else None
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        add_user_file(filepath, tag="userfile")
        return jsonify({
            "success": True,
            "message": f"File '{file.filename}' added to knowledge base."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Serve audio files from root/static/audio ---
@app.route("/static/audio/<path:filename>")
def serve_audio(filename):
    """Serve audio files from the root static/audio folder."""
    return send_from_directory(AUDIO_ROOT, filename)


if __name__ == "__main__":
    app.run(debug=True)
