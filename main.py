from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


origins = [
    "http://localhost",  # Frontend served from localhost (for dev mode)
    "http://localhost:3000",  # Frontend running on port 3000 (if using React, for example)
    "http://localhost:8001",  # Any other ports if needed
    "*",  # Allow all origins (use cautiously in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Adjust origins here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Your route setup
@app.get("/ping")
async def ping():
    return "Hello, I am alive"


MODEL = tf.keras.models.load_model("../models/1.keras")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the file into a numpy array
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)