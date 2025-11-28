# 🤖 MkDocs RAG Assistant

A "Retrieval-Augmented Generation" (RAG) AI application that answers technical questions about [MkDocs](https://www.mkdocs.org/). It uses Google's Gemini models to understand queries and retrieves accurate information from a local vector database created from the MkDocs user guide.

## 🚀 Features

* **RAG Architecture**: Retrieves specific context from documentation rather than relying on general knowledge.
* **Smart Chunking**: Uses `MarkdownHeaderTextSplitter` to respect the hierarchy of technical documentation (keeping configuration options with their headers).
* **Strict Mode**: The AI is prompted to strictly admit when it doesn't know an answer, preventing hallucinations.
* **Source Citations**: The UI displays exactly which source files were used to generate the answer.
* **Streamlit GUI**: A clean, chat-based web interface with history.
* **Persistent Database**: Uses ChromaDB to store embeddings locally, so you only need to process data once.

## 📂 Project Structure

| File / Folder | Description |
| :--- | :--- |
| **`app.py`** | The main entry point. Runs the Streamlit web application and Chat UI. |
| **`RAG.py`** | The backend logic. Handles database connections, context retrieval, and LLM generation. |
| **`Embedding.ipynb`** | The data pipeline. Reads Markdown files, cleans them, chunks them, and saves them to the Vector DB. |
| **`user-guide/`** | Folder containing the raw MkDocs Markdown documentation files. |
| **`mkdocs_db/`** | (Generated) The local ChromaDB folder storing the vector embeddings. |
| **`.env`** | (Not in repo) Configuration file for storing the `GOOGLE_API_KEY`. |
| **`requirements.txt`** | List of Python dependencies required to run the project. |

## 🛠️ Prerequisites

* Python 3.10 or higher.
* A Google Cloud API Key (with access to Gemini models).

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/mkdocs-rag-assistant.git](https://github.com/YOUR_USERNAME/mkdocs-rag-assistant.git)
    cd mkdocs-rag-assistant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Create a `.env` file in the root directory and add your API Key:
    ```ini
    GOOGLE_API_KEY=AIzaSy...YourKeyHere
    ```

## 🏃 Usage

### Step 1: Ingest Data (Run Once)
Before running the app, you must process the documentation and populate the database.
1.  Open `Embedding.ipynb` in VS Code or Jupyter.
2.  Run all cells.
3.  *Result:* A folder named `mkdocs_db` will be created.

### Step 2: Run the Application
Launch the web interface:

```bash
python app.py
