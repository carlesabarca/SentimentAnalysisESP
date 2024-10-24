import gradio as gr
from transformers import pipeline
from PyPDF2 import PdfReader

# Inicializar el modelo de análisis de sentimiento con DistilBETO
sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis")

def analyze_text(text):
    result = sentiment_analysis(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Label: {label}, Score: {score}"

def analyze_pdf(pdf):
    # Leer el contenido del PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    # Análisis del contenido del PDF
    result = sentiment_analysis(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Label: {label}, Score: {score}"

# Interfaz de Gradio con dos inputs: texto o PDF
with gr.Blocks() as demo:
    gr.Markdown("# Análisis de Sentimientos con DistilBETO")
    gr.Markdown("Ingrese un texto en español o cargue un PDF para analizar su sentimiento usando DistilBETO.")
    
    with gr.Row():
        text_input = gr.Textbox(label="Texto", placeholder="Escribe aquí el texto para análisis")
        pdf_input = gr.File(label="Cargar PDF", type="file")
    
    output = gr.Textbox(label="Resultado")
    
    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=analyze_text, inputs=text_input, outputs=output)
    submit_btn.click(fn=analyze_pdf, inputs=pdf_input, outputs=output)

demo.launch()