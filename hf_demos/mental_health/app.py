# app.py
import gradio as gr
import pickle

# Load saved models (no re-training needed)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model_tree.pkl", "rb") as f:
    model_tree = pickle.load(f)

with open("model_nn.pkl", "rb") as f:
    model_nn = pickle.load(f)

# Prediction function
def predict(text, model_choice):
    if not text.strip():
        return "⚠️ Please enter a statement.", ""

    text_vec = vectorizer.transform([text])

    if model_choice == "Neural Network (93.58% accuracy)":
        model = model_nn
        model_name = "Neural Network"
    else:
        model = model_tree
        model_name = "Decision Tree"

    prediction = model.predict(text_vec)[0]
    proba = model.predict_proba(text_vec)[0]
    confidence = f"{max(proba) * 100:.1f}%"

    label = "🟢 Normal" if prediction == 1 else "🔴 Depression"

    result = f"**{label}**"
    detail = f"Model: {model_name} | Confidence: {confidence}"

    return result, detail

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="Mental Health Sentiment Analysis") as demo:

    gr.Markdown("""
    # 🧠 Mental Health Sentiment Analysis
    Classifies mental health statements as **Normal** or **Depression**
    using models trained on 53,000+ real-world statements.
    """)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter a statement",
                placeholder="e.g. I can't stop worrying about everything...",
                lines=4
            )
            model_choice = gr.Radio(
                choices=[
                    "Neural Network (93.58% accuracy)",
                    "Decision Tree (90.78% accuracy)"
                ],
                value="Neural Network (93.58% accuracy)",
                label="Choose Model"
            )
            submit_btn = gr.Button("Analyse", variant="primary")

        with gr.Column():
            result_output = gr.Markdown(label="Prediction")
            detail_output = gr.Textbox(label="Details", interactive=False)

            gr.Markdown("""
            ### 📊 Model Info
            | Model | Accuracy |
            |-------|----------|
            | Neural Network (MLP) | 93.58% |
            | Decision Tree | 90.78% |
            """)

    # Example inputs
    gr.Examples(
        examples=[
            ["I can't stop worrying about everything", "Neural Network (93.58% accuracy)"],
            ["Today has been amazing, I feel so happy!", "Neural Network (93.58% accuracy)"],
            ["Even the smallest things feel like too much right now", "Decision Tree (90.78% accuracy)"],
            ["I've been working hard and seeing results makes me feel fulfilled", "Decision Tree (90.78% accuracy)"],
            ["I can't stop smiling, life is good", "Neural Network (93.58% accuracy)"],
        ],
        inputs=[text_input, model_choice]
    )

    submit_btn.click(
        fn=predict,
        inputs=[text_input, model_choice],
        outputs=[result_output, detail_output]
    )

    gr.Markdown("""
    ---
    > ⚠️ **Disclaimer:** This tool is for educational purposes only and is not a substitute
    for professional mental health diagnosis or advice.

    Built by [Syahmi Amin](https://sabinshamsul.dev) · 
    [Portfolio](https://sabinshamsul.dev) · 
    [GitHub](https://github.com/sabinshamsul)
    """)

demo.launch()