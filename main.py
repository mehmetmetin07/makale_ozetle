from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

# Model yükle
model_name = "ozcangundes/mt5-small-turkish-summarization"

class Text(BaseModel):
    content: str

@app.post("/summarize")
async def summarize_text(text: Text):
    # Pipeline oluştur
    summarizer = pipeline("summarization", model=model_name)
    
    # Özet oluştur
    summary = summarizer(text.content,
                        max_length=150,
                        min_length=50,
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True)
    return {"summary": summary[0]['summary_text']}

@app.get("/")
def read_root():
    return {"status": "API is running"}