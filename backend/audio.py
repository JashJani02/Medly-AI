import os
import time
from gtts import gTTS

AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def generate_audio(text, lang="en"):

    if not text or text.strip() == "":
        return None

    filename = f"reply_{int(time.time())}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    tts = gTTS(text=text, lang=lang)
    tts.save(filepath)

    return filepath
