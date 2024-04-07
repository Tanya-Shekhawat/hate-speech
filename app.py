from typing import Union

from fastapi import FastAPI
from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Load model and tokenizer
pipe = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english")
tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")


@app.post("/classify/")
async def classify_text(text: str):
    # Process the input text using the pipeline
    result = pipe(text)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
