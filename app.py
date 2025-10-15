# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json, numpy as np, requests, io, os
from PIL import Image
from model import load_model, image_to_embedding, cosine_similarity
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Visual Product Matcher")

# Absolute path to your dataset folder
DATASET_PATH = r"E:\visual product\dataset"

# Mount dataset folder to serve images
app.mount("/dataset", StaticFiles(directory=DATASET_PATH), name="dataset")

# Load products.json
with open(r"E:\visual product\products.json", "r") as f:
    products = json.load(f)

# Precompute product embeddings
product_embeddings = np.array([p["embedding"] for p in products])

# Load your model
model = load_model()

# ---------------- Frontend HTML ----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Visual Product Matcher</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            .container { max-width: 600px; margin: auto; }
            img { max-width: 200px; margin: 10px; border: 1px solid #ccc; }
            .product { display: inline-block; text-align: center; margin: 10px; }
            input, button { padding: 8px; margin: 5px 0; width: 100%; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Visual Product Matcher</h2>
            <label>Upload Image:</label>
            <input type="file" id="fileInput"><br>
            <label>Or Image URL:</label>
            <input type="text" id="urlInput" placeholder="Paste image URL"><br>
            <label>Number of results:</label>
            <input type="number" id="topK" value="5"><br>
            <button onclick="searchImage()">Search</button>

            <h3>Uploaded Image:</h3>
            <img id="uploadedImg" src="" alt="" />

            <h3>Similar Products:</h3>
            <div id="results"></div>
        </div>

        <script>
            async function searchImage() {
                const fileInput = document.getElementById("fileInput");
                const urlInput = document.getElementById("urlInput").value;
                const topK = document.getElementById("topK").value;
                const resultsDiv = document.getElementById("results");
                const uploadedImg = document.getElementById("uploadedImg");

                resultsDiv.innerHTML = "";

                const formData = new FormData();
                if (fileInput.files.length > 0) {
                    formData.append("file", fileInput.files[0]);
                    uploadedImg.src = URL.createObjectURL(fileInput.files[0]);
                } else if (urlInput) {
                    formData.append("image_url", urlInput);
                    uploadedImg.src = urlInput;
                } else {
                    alert("Please upload a file or enter an image URL.");
                    return;
                }
                formData.append("top_k", topK);

                try {
                    const response = await fetch("/search", {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();

                    if (data.error) {
                        alert(data.error);
                        return;
                    }

                    data.results.forEach(p => {
                        const div = document.createElement("div");
                        div.className = "product";
                        div.innerHTML = `
                            <img src="${p.image_path}" alt="${p.name}" />
                            <div><strong>${p.name}</strong></div>
                            <div>${p.category}</div>
                            <div>Score: ${p.score.toFixed(2)}</div>
                        `;
                        resultsDiv.appendChild(div);
                    });
                } catch (err) {
                    console.error(err);
                    alert("Failed to fetch results.");
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ---------------- Search Endpoint ----------------
@app.post("/search")
async def search(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    top_k: int = Form(10)
):
    if not file and not image_url:
        return {"error": "Please upload an image or provide an image URL."}

    # Load uploaded image
    if file:
        img = Image.open(file.file).convert("RGB")
    else:
        try:
            response = requests.get(image_url, timeout=5)
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to load image from URL: {str(e)}"}

    # Generate embedding
    emb = image_to_embedding(model, img)

    # Compute similarity
    sims = cosine_similarity(emb, product_embeddings)
    idx = np.argsort(-sims)[:top_k]

    # Prepare results
    results = []
    for i in idx:
        p = products[int(i)]
        # Convert absolute path to relative path for frontend
        rel_path = os.path.relpath(p["image_path"], DATASET_PATH).replace("\\", "/")
        dataset_url = f"/dataset/{rel_path}"
        results.append({
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "image_path": dataset_url,
            "score": float(sims[int(i)])
        })

    return {"results": results}


# ---------------- Run Server ----------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
