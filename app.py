import gradio as gr
from transformers import pipeline
import PyPDF2

# Cargar el modelo de análisis de sentimientos
sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_text(input_text):
    result = sentiment_analysis(input_text)[0]
    label = result['label']
    score = round(result['score'] * 100, 2)  # Convertir a porcentaje

    # Definir la categoría de sentimiento según la etiqueta
    if label == "1 star":
        sentiment_label = "Muy negativo"
    elif label == "2 stars":
        sentiment_label = "Negativo"
    elif label == "3 stars":
        sentiment_label = "Neutro"
    elif label == "4 stars":
        sentiment_label = "Positivo"
    else:
        sentiment_label = "Muy positivo"

    return f"{sentiment_label} ({score}%)", score

def analyze_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return analyze_text(text)

# Interfaz de Gradio
def text_analysis(input_text):
    sentiment, score = analyze_text(input_text)
    return sentiment, score

def pdf_analysis(pdf_file):
    sentiment, score = analyze_pdf(pdf_file)
    return sentiment, score

with gr.Blocks() as app:
    gr.Markdown("## Análisis de Sentimientos con DistilBETO")

    with gr.Column():
        gr.Markdown("### Análisis de Texto")
        text_input = gr.Textbox(label="Ingrese texto para análisis")
        text_button = gr.Button("Analizar texto")
        text_output = gr.Label(label="Resultado del análisis")
        slider_output = gr.Slider(label="Nivel de sentimiento", minimum=0, maximum=100, interactive=False)

        text_button.click(text_analysis, inputs=text_input, outputs=[text_output, slider_output])

    with gr.Column():
        gr.Markdown("### Análisis de PDF")
        pdf_input = gr.File(label="Subir PDF para análisis", file_types=[".pdf"])
        pdf_button = gr.Button("Analizar PDF")

        pdf_button.click(pdf_analysis, inputs=pdf_input, outputs=[text_output, slider_output])

# Ejecutar la aplicación
app.launch()