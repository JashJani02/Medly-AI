import os
import streamlit as st
from backend.llm import generate_response
from backend.audio import generate_audio
from backend.rag import retrieve_context, add_user_file
from utils.prompts import build_medical_prompt


def show_chat_interface():

    patient = st.session_state.patient_data

    st.success(f"Welcome {patient['name']} 👋")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "audio" in msg and msg["audio"]:
                if os.path.exists(msg["audio"]):
                    with open(msg["audio"], "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")

    prompt = st.chat_input("Describe your symptoms...")

    if prompt:

        # show user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG retrieval
        context = retrieve_context(prompt)


        if not context.strip():
            context= "No specific medical documents provided. Use general medical knowledge"

        # Build prompt
        full_prompt = build_medical_prompt(patient, prompt, context)

        # Generate response
        with st.spinner("Analyzing symptoms..."):
            response = generate_response(full_prompt)

        # Generate audio
        audio_path = generate_audio(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "audio": audio_path
        })

        with st.chat_message("assistant"):
            st.markdown(response)

            if audio_path and os.path.exists(audio_path):
                with open(audio_path,"rb") as audio_file:
                    st.audio(audio_file.read(),format="audio/mp3")

    with st.sidebar:
        st.subheader("Context files:")
        uploaded_file = st.file_uploader("Upload medical report (.pdf/.txt)", type=["pdf", "txt"])

        # Check if file is uploaded before trying to use it
        if uploaded_file is not None:
            save_path = os.path.join("data", "uploads", uploaded_file.name)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Use session_state to prevent re-processing on every page interaction
            if st.session_state.get("last_file") != uploaded_file.name:
                with open(save_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())
                
                with st.spinner("Indexing docs..."):
                    add_user_file(save_path)
                    st.session_state.last_file = uploaded_file.name
                    st.success(f"Uploaded {uploaded_file.name} as Context")