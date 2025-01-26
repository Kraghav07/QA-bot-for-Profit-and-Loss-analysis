import streamlit as st
import os
import cohere
import faiss
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
from dotenv import load_dotenv
import pandas as pd

# Load environment variables (for Cohere API key)
load_dotenv()

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Streamlit UI
st.set_page_config(page_title="QA Assistant", layout="wide", page_icon="💼")

# Page Header
st.title("💬 P&L 2024 Insights Assistant")
st.markdown(
    """
    Welcome to the P&L 2024 Insights Assistant! Upload a PDF document, ask questions, and get instant answers based on the content. 
    Leverage **AI-powered embeddings** to search for relevant sections of your document.
    """
)

# Sidebar for navigation and upload
st.sidebar.header("🔧 Settings & Tools")
st.sidebar.markdown("Use the tools below to upload files and manage queries.")

# Tabs for different features
tabs = st.tabs(["📂 File Upload", "❓ Query", "📝 History"])

# Initialize history storage
if 'history' not in st.session_state:
    st.session_state.history = []

# Tab 1: File Upload
with tabs[0]:
    st.header("📂 Upload Your PDF")
    uploaded_file = st.file_uploader("Upload a PDF file to extract insights:", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing your file..."):
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.read())
            
            # Function to read and process PDF
            def read_pdf(file_path):
                file_loader = PyPDFLoader(file_path)
                documents = file_loader.load()
                return documents
            
            # Load and chunk the PDF
            doc = read_pdf("temp_pdf.pdf")
            
            def chunk_data(docs, chunk_size=800, chunk_overlap=50):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                docs_chunked = text_splitter.split_documents(docs)
                return docs_chunked
            
            documents = chunk_data(docs=doc)
            st.success(f"Uploaded and processed: `{uploaded_file.name}`")

# Cohere Embedding Class
class CohereEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = co.embed(texts=texts)
        return response.embeddings
    
    def embed_query(self, text: str) -> list[float]:
        response = co.embed(texts=[text])
        return response.embeddings[0]

# FAISS Embeddings
def store_faiss_embeddings(embeddings, documents):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

if uploaded_file:
    texts = [doc.page_content for doc in documents]
    cohere_embedder = CohereEmbeddings()
    embeddings = cohere_embedder.embed_documents(texts)
    faiss_index = store_faiss_embeddings(embeddings, documents)

    def search_faiss_index(query):
        query_embedding = cohere_embedder.embed_query(query)
        query_vector = np.array([query_embedding])
        D, I = faiss_index.search(query_vector, k=5)
        return I

# Tab 2: Query
with tabs[1]:
    st.header("❓ Ask Your Question")
    query = st.text_input("Type your question about Budget 2024:", placeholder="e.g., What is the allocation for health?")
    if query and uploaded_file:
        with st.spinner("Searching and processing your query..."):
            result_indices = search_faiss_index(query)
            
            # Run Question Answering Chain on matched documents
            llm = Cohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            matching_documents = [documents[i] for i in result_indices[0]]
            answer = qa_chain.run(input_documents=matching_documents, question=query)
            
            # Display the answer
            st.markdown("### 📖 Answer:")
            st.success(answer)
            
            # Save to history
            st.session_state.history.append({"query": query, "answer": answer})

# Tab 3: History
with tabs[2]:
    st.header("📝 Query History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history.clear()
            st.experimental_rerun()
    else:
        st.info("No queries yet. Start asking questions!")

# Footer
st.markdown("---")
st.markdown("🔍 Powered by **Streamlit** and **Cohere** | Made with ❤️ for QA 2024 insights.")
