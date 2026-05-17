import gradio as gr
import json
from src.model.llm.load_llm import generate_text
from src.file_loader.pdf_loader import compute_features, prediction_data
from src.model.core.load_model import predict
import shutil


def process_pdf(file):
    save_path = "loan.pdf"
    if file is None:
         return "", "Please upload a PDF file.", *[None]*5
    shutil.copy(file.name, save_path)

    try:
        data = prediction_data(save_path)

        if all(values is None for values in data.values()):
            return "Invalid PDF", "Could not extract meaningful data.", *[None]*5

        df = compute_features(data)

        if df.isnull().mean().mean() > 0.8:
            return "Invalid PDF", "Too many missing values.", *[None]*5

        prediction, predict_data, figs = predict(df)

    except Exception as e:
        return "Invalid PDF", f"Processing failed: {e}", *[None]*5

    # Load prompts
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

    while len(figs) < 5:
        figs.append(None)

    return f"{prediction:.2f}", explanation, *figs[:5]

def reset_outputs(file):
    if file is None:
        return "", "Please upload a PDF file.", *[None]*5
    return "", "File uploaded. Click 'Analyze' to process.", *[None]*5

# UI
with gr.Blocks(title="Loan Risk Analyzer") as app:
    gr.Markdown("# 🏦 Loan Risk Analyzer")
    gr.Markdown("Upload a loan application PDF to get prediction and SHAP analysis.")

    with gr.Row():
        pdf_input = gr.File(label="📤 Upload Loan PDF", type="filepath")
        submit_btn = gr.Button("Analyze", variant="primary")

    gr.Markdown("---")

    with gr.Row():
        risk_score = gr.Textbox(label="📊 Risk Score", interactive=False)
        explanation = gr.Textbox(label="💬 Explanation", lines=6, interactive=False)

    gr.Markdown("---")
    gr.Markdown("## 🔍 Top SHAP Feature Analysis")

    # ✅ CORRECT ROW-WISE LAYOUT (NO render())
    figs = []

    for i in range(0, 5, 3):  # 3 per row
        with gr.Row():
            for j in range(i, min(i+3, 5)):
                fig = gr.Plot(label=f"Feature {j+1}")
                figs.append(fig)

    submit_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[risk_score, explanation, *figs]
    )

    pdf_input.change(
        fn=reset_outputs,
        inputs=pdf_input,
        outputs=[risk_score, explanation, *figs]
    )


app.launch(css=""".gradio-container {
        max-width: 1400px !important;
        margin: auto;
    }
    .gr-row {
        flex-wrap: nowrap !important;
    }
    """)