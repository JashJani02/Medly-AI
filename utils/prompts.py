def build_medical_prompt(patient,symptoms,context=""):

    patient_context = f"""
Patient Information:
Name: {patient['name']}
Age: {patient['age']}
Country: {patient['country']}
State: {patient['state']}
Allergies: {patient['allergies']}
"""
    
    prompt = f"""

You are a helpful and empathetic medical assistant.

{patient_context}

Relevant medical context:
{context}

User symptoms:
{symptoms}

Provide the answer using this structure:

Possible Causes:
- List 2-3 common explanations.

Preventive Measures:
- Suggest lifestyle or health precautions.

Medical Advice:
- Suggest safe next steps.

Disclaimer:
This is general health information, not a medical diagnosis.
"""
    
    return prompt