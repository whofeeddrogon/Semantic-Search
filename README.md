# Semantic Search Engine

A modern semantic and keyword search engine built with Streamlit, FastAPI, and Qdrant. It is designed to be completely modular and deployable via Docker Compose.

## 🚀 Features

* **Advanced Search Interface:** A beautiful, responsive UI built with Streamlit indicating product prices, discounts, and real-time similar item matching.
* **Hybrid Search Ready:** Designed to handle both Semantic (Dense) vectors and Keyword (Sparse / BM25) vectors.
* **Qdrant Vector Database:** Includes a robust integration with Qdrant for storing and querying embeddings instantly.
* **FastAPI Backend:** A lightweight, high-performance API that bridges the UI and the vector database.
* **TEI Integration (Optional):** Pre-configured to easily link with Hugging Face's `text-embeddings-inference` (TEI) server for blisteringly fast embedding computations.

## 📋 Architecture & Tech Stack

1. **Frontend:** Streamlit (`streamlit_app.py`)
2. **Backend:** FastAPI (`app.py`)
3. **Database:** Qdrant (`qdrant_storage/`)
4. **Embeddings Engine:** SentenceTransformers (Dense) & FastEmbed (Sparse), with Docker Compose support for TEI.

## 🛠️ Local Development

### Prerequisites

* Python 3.10+
* Docker & Docker Compose (for Qdrant)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/whofeeddrogon/Semantic-Search.git
   cd Semantic-Search
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Qdrant database:**
   ```bash
   docker-compose up -d qdrant
   ```

4. **Start the FastAPI backend:**
   ```bash
   python app.py
   ```
   *The API will be available at `http://localhost:8000`*

5. **Start the Streamlit UI:**
   Open a new terminal and run:
   ```bash
   streamlit run streamlit_app.py
   ```
   *The UI will be available at `http://localhost:8501`*

## 🐳 Deployment (Coolify / Docker Compose)

This project has been specifically structured for one-click deployment via **Coolify**.

1. Create a **New Resource** -> **Git Repository** in Coolify.
2. Select your repository. Coolify will automatically detect the `docker-compose.yml`.
3. Hit **Deploy**. 

The Docker Compose configuration will spin up:
* The **Qdrant** server
* The **TEI (Text Embeddings Inference)** server for remote, high-speed neural network evaluations (`BAAI/bge-m3` model).

*Note: You may need to adapt your `API_URL` environment variables if you deploy the frontend and backend onto containerized instances down the line.*

## 📂 Project Structure

* `app.py`: FastAPI application serving the `/search` endpoint.
* `streamlit_app.py`: Streamlit frontend application.
* `search_engine.py`: Defines search modes (Dense/Sparse/Hybrid) and queries the database via the model wrapper.
* `database_manager.py`: Qdrant client utility methods (upsert, search).
* `model_loader.py`: Singleton loaders for SentenceTransformers (Dense) and FastEmbed (Sparse).
* `docker-compose.yml`: Qdrant and TEI deployment file.
* `config.py`: Configuration parameters (Model name, database URL, etc.)
* `qdrant_storage/`: Ignored via `.gitignore`. The local bind-mount folder holding the Qdrant vector database.

## 🔧 Search Modes

You can toggle between the following modes directly in the UI under **Search Settings**:
1. **Keyword Search (BM25):** Fast, exact text matching based on word frequencies.
2. **Semantic Search (Vector):** Context-aware, conceptual matching using neural network embeddings. 