import gradio as gr
import json
from src.model.llm.load_llm import generate_text
from src.file_loader.pdf_loader import compute_features, prediction_data
from src.model.core.load_model import predict
import shutil
import os

def process_pdf(file):
    save_path = "loan.pdf"
    shutil.copy(file.name, save_path)
    try:
        data = prediction_data(save_path)
        if all(values is None for values in data.values()):
            return f"Please upload a valid PDF file.", f"Could not extract any meaningful data from PDF."

        df = compute_features(data)

        if df.isnull().mean().mean() > 0.5:
            return f"Please upload a valid PDF file.", f"Too many missing values in extracted data. Could not process PDF."

        prediction, predict_data = predict(df)

    except Exception as e:
        return f"Please upload a valid PDF file.", f"The file could not be processed." 


    with open("src/model/llm/SYSTEM_PROMPT.txt", "r") as f:
        SYSTEM_PROMPT = f.read().strip()

    with open("src/model/llm/corr.json", "r") as f:
        corr = json.load(f)

    user_input = f"""
    Feature Correlations with RiskScore:
    {corr}

    Applicant Data:
    {predict_data}

    Predicted RiskScore:
    {prediction:.2f}

    Explain the reason for this prediction.
    """

    explanation = generate_text(user_input, SYSTEM_PROMPT)

    return f"{prediction:.2f}", explanation


app = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload Loan PDF"),
    outputs=[
        gr.Textbox(label="Risk Score"),
        gr.Textbox(label="Explanation")
    ],
    title="Loan Risk Analyzer",
    description="Upload a loan application PDF to get prediction + explanation"
)

