import gradio as gr
from transformers import pipeline

# Cargar el modelo DistilBETO para análisis de sentimientos
modelo = "finiteautomata/beto-sentiment-analysis"
sentiment_analysis = pipeline("sentiment-analysis", model=modelo, tokenizer=modelo)

def analizar_sentimiento(texto):
    resultado = sentiment_analysis(texto)[0]
    label = resultado['label']
    score = resultado['score']

    # Determinar si es positivo, negativo o neutro
    if label == 'NEG':
        sentimiento = "Negativo"
    elif label == 'NEU':
        sentimiento = "Neutro"
    else:
        sentimiento = "Positivo"

    return f"Sentimiento: {sentimiento}, Confianza: {score:.2f}"

# Interfaz Gradio
demo = gr.Interface(
    fn=analizar_sentimiento,
    inputs="text",
    outputs="text",
    title="Análisis de Sentimientos con DistilBETO",
    description="Ingrese un texto en español para analizar su sentimiento usando DistilBETO."
)

demo.launch()