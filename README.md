# Intelligent HR RAG Assistant (Streamlit)

This app converts the core pipeline from the supplied Kaggle notebook into a Streamlit HR policy chatbot.

## Features
- Upload multiple HR policy PDFs
- Extract PDF text with PyPDF
- Chunk text using 1000 characters and 150-character overlap
- Embed with `sentence-transformers/all-MiniLM-L6-v2`
- Retrieve with FAISS inner-product similarity over normalized embeddings
- Generate grounded answers with Groq `openai/gpt-oss-120b`
- Display retrieved source documents

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create `.streamlit/secrets.toml`:

   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key"
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Upload the HR policy PDFs in the sidebar and click **Process Documents**.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `.gitignore` to a GitHub repository.
2. Do not upload `.streamlit/secrets.toml` or expose your API key.
3. Create a new app in Streamlit Community Cloud and select the repository and `app.py`.
4. In the app settings, add this secret:

   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key"
   ```

5. Deploy, open the app URL, upload the PDFs, and process them.

## Important notes

- The Kaggle notebook's `/kaggle/input/...` path is specific to Kaggle. This app uses PDF uploads instead.
- The PDFs are processed in the running app session. If the app restarts, upload and process them again.
- Scanned/image-only PDFs may require OCR; PyPDF extracts selectable PDF text.
- The notebook's Kaggle test CSV and submission-generation cells are not part of the interactive Streamlit chatbot.
