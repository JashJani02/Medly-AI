import streamlit as st

def show_patient_form():

    st.subheader("Patient Information")

    allergic = st.toggle("Do you have allergies")

    with st.form("patient_form"):

        name = st.text_input("Name: ")
        age = st.number_input("Age", min_value=0, max_value=110)
        country = st.text_input("Country")
        state = st.text_input("State")
        

        allergies = ""

        if allergic:
            allergies = st.text_input("List your allergies")

        submitted = st.form_submit_button("Submit")

        if submitted:

            st.session_state.patient_data = {
                "name": name,
                "age": age,
                "country": country,
                "state": state,
                "allergies": allergies if allergic else "None"
            }

            st.session_state.form_submitted = True

            st.rerun()