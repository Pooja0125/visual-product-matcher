import numpy as np
import tensorflow as tf
from PIL import Image

def load_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3), include_top=False, pooling="avg", weights="imagenet"
    )
    return base

def preprocess(img: Image.Image):
    img = img.resize((224,224))
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr,0)

def image_to_embedding(model, pil_img):
    x = preprocess(pil_img)
    emb = model.predict(x)
    emb = emb[0] / (np.linalg.norm(emb[0]) + 1e-10)
    return emb

def cosine_similarity(query_emb, embeddings):
    q = query_emb / (np.linalg.norm(query_emb)+1e-10)
    embs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True)+1e-10)
    return np.dot(embs, q)
