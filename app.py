import gradio as gr
from transformers import pipeline

# Cargar el modelo DistilBETO
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Definir la función de análisis de sentimientos
def analyze_sentiment(text):
    results = sentiment_analysis(text)
    return f"Label: {results[0]['label']}, Score: {results[0]['score']}"

# Configurar la interfaz de Gradio
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Análisis de Sentimientos con DistilBETO",
    description="Ingrese un texto en español para analizar su sentimiento."
)

# Lanzar la aplicación
demo.launch()