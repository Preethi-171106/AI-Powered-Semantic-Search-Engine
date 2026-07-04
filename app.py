import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Powered Semantic Search Engine",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Powered Semantic Search Engine")
st.write("Welcome to the AI Semantic Search Engine!")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
# ---------------- GEMINI ----------------

load_dotenv()

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

print("API KEY =", api_key)

genai.configure(api_key=api_key)

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # ---------------- READ PDF ----------------

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("✅ PDF uploaded successfully!")

    # ---------------- SHOW TEXT ----------------

    st.subheader("📄 Extracted Text")
    st.write(text[:1000])

    # ---------------- CHUNKING ----------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    st.success(f"✅ Total Chunks Created: {len(chunks)}")

    st.subheader("📄 First Chunk")
    st.write(chunks[0])

    # ---------------- EMBEDDINGS ----------------

    embeddings = model.encode(chunks)

    st.success(f"✅ Embeddings Created: {len(embeddings)}")

    # ---------------- FAISS ----------------

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    st.success(f"✅ FAISS Index Created with {index.ntotal} Chunks")

    # ---------------- SEARCH ----------------

    st.subheader("🔍 Ask a Question")

    query = st.text_input(
        "Enter your question about the PDF"
    )

    if st.button("Search", key="search_button"):

        if query.strip() == "":
            st.warning("Please enter a question.")

        else:
            query_embedding = model.encode([query])
            query_embedding = np.array(query_embedding).astype("float32")
            
            distances, indices = index.search(
                query_embedding,
                k=3
            )
            st.success("🎯 Top 3 Matching Results")
            for rank, idx in enumerate(indices[0]):
                
                st.markdown("---")
                
                if rank == 0:
                    title = "🥇 Best Match"
                elif rank == 1:
                    title = "🥈 Second Best Match"
                else:
                    title = "🥉 Third Best Match"

                score = float(max(0, min(100, 100 - float(distances[0][rank]))))

                st.subheader(title)

                st.write(chunks[idx])

                st.progress(float(score / 100))

                st.caption(f"📊 Similarity Score : {score:.2f}%")

                # Best matching chunk
            best_chunk = chunks[indices[0][0]]

            prompt = f"""
            You are an AI assistant.
                
            Answer the user's question only using the following PDF content.
                
            Question:
            {query}
                
            PDF Content:
            {best_chunk}
                
            Give a short and clear answer.
            """
            response = gemini_model.generate_content(prompt)
                    
            st.success("✅ Answer Generated Successfully!")

            st.subheader("🤖 AI Generated Answer")
            st.write(response.text)

            st.download_button(
                
                label="📥 Download AI Answer",
                data=response.text,
                file_name="AI_Answer.txt",
                mime="text/plain"
            )

                  


            
            