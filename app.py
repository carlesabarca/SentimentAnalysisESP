import gradio as gr
from transformers import pipeline
from PyPDF2 import PdfReader
import logging

# Imprimir la versión de Gradio en los logs
logging.info(f"Versión de Gradio instalada: {gr.__version__}")

# Cargar el modelo de DistilBETO
sentiment_analysis = pipeline("sentiment-analysis", 
                              model="nlptown/bert-base-multilingual-uncased-sentiment")

# Función para analizar texto
def analyze_text_sentiment(text):
    result = sentiment_analysis(text)[0]
    score = result['score']
    label = result['label']
    
    # Convertir la etiqueta de sentimiento en un valor para el slider y su literal
    if label == "1 star":
        slider_value = 0
        literal = "Muy negativo"
    elif label == "2 stars":
        slider_value = 25
        literal = "Negativo"
    elif label == "3 stars":
        slider_value = 50
        literal = "Neutro"
    elif label == "4 stars":
        slider_value = 75
        literal = "Positivo"
    elif label == "5 stars":
        slider_value = 100
        literal = "Muy positivo"

    # Añadir el nivel de confianza al literal
    confidence = round(score * 100, 2)
    literal_with_confidence = f"{literal} (Confianza: {confidence}%)"

    return slider_value, literal_with_confidence

# Función para extraer texto de PDF y analizar
def analyze_pdf_sentiment(pdf_file):
    pdf_reader = PdfReader(pdf_file.name)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return analyze_text_sentiment(text)

# Interfaz de Gradio
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Texto"):
            text_input = gr.Textbox(label="Ingrese su texto aquí", lines=5)
            analyze_button = gr.Button("Analizar")
            slider_output = gr.Slider(label="Nivel de Sentimiento", minimum=0, maximum=100, step=1, interactive=False)
            label_output = gr.Label()

            analyze_button.click(analyze_text_sentiment, inputs=text_input, outputs=[slider_output, label_output])

        with gr.Tab("PDF"):
            pdf_input = gr.File(label="Subir PDF", file_types=[".pdf"])
            analyze_button = gr.Button("Analizar")
            slider_output = gr.Slider(label="Nivel de Sentimiento", minimum=0, maximum=100, step=1, interactive=False)
            label_output = gr.Label()

            analyze_button.click(analyze_pdf_sentiment, inputs=pdf_input, outputs=[slider_output, label_output])

demo.launch()