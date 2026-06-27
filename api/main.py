from fastapi import FastAPI
import tensorflow as tf
import uvicorn
import os

from responses import NewsItem

app = FastAPI(name="Fake News Detection API")

MODEL_PATH = "model/fake_news_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict")
def predict_news(item: NewsItem):
    text = item.text
    prediction = model.predict(tf.constant([[text]]))
    score = float(prediction[0][0])
    is_fake = bool(score > 0.5)
    confidence = float(score) if is_fake else float(1 - score)
    return {
        "is_fake": is_fake,
        "confidence": round(confidence, 4),
        "score": score
    }

if (__name__ == "__main__"):
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)