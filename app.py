import streamlit as st
from components.form import show_patient_form
from components.chat_interface import show_chat_interface

st.set_page_config(page_title="Medly-Ai",page_icon="🩺",layout="wide")

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

st.title("Medly-Ai: Your Personal Medical Assistant")

if not st.session_state.form_submitted:
    show_patient_form()

else:
    show_chat_interface()