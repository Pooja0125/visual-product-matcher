"Visual Product Matcher"

# Overview
Visual Product Matcher is a web application that allows users to find visually similar products by uploading an image or providing an image URL. The application uses machine learning to compute image similarity and returns a list of products from the dataset that closely match the query.

# Features
- Upload an image file or provide an image URL.
- View the uploaded image on the interface.
- Retrieve and display visually similar products.
- Filter results by similarity score.
- Mobile responsive design.
- Clean, production-ready code with basic error handling.

# Dataset
- The dataset contains 50+ products with images.
- Each product has basic metadata: name, category, and image path.
- Dataset images are served as static files within the application.

# Tech Stack
- **Backend:** FastAPI, Python
- **Machine Learning:** TensorFlow, NumPy, cosine similarity
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Local or cloud hosting (optional)

# How to Run Locally
1. Clone the repository:
```bash
git clone https://github.com/Pooja0125/visual-product-matcher.git

Navigate to the backend folder:
cd visual-product-matcher/backend

Install required dependencies:
pip install fastapi uvicorn tensorflow numpy pillow requests

Run the FastAPI application:
uvicorn app:app --reload --port 8000

Open your browser and go to:
http://127.0.0.1:8000

Use the app:
Upload an image file or provide an image URL in the search field.
View the uploaded image.
See the list of visually similar products along with similarity scores.
