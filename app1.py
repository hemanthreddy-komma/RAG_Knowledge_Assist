import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Personal AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    "<h1 style='text-align: center;'>🤖 Personal AI Assistant</h1>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<h3 style='text-align: center;'>Assistant Console</h3>",
    unsafe_allow_html=True,
)

if not GROQ_API_KEY:
    GROQ_API_KEY = st.sidebar.text_input(
        "Groq API Key", type="password",
        help="Get a free key at https://console.groq.com/keys. Set GROQ_API_KEY in a .env file to skip this.",
    )

if not GROQ_API_KEY:
    st.warning("Please provide a Groq API key to continue (free at console.groq.com/keys).")
    st.stop()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know.\n\n{context}",
        ),
        ("human", "{input}"),
    ]
)
qa_document_chain = create_stuff_documents_chain(llm, qa_prompt)

# -----------------------------
# Session state
# -----------------------------
for key in ["url_vectorstore", "pdf_vectorstore", "pdf_bytes", "pdf_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=1000,
    chunk_overlap=200,
)

# =============================
# Sidebar: data source selection
# =============================
st.sidebar.header("Data Source")
source_type = st.sidebar.radio("Select Source", ["URLs", "PDF"])

# -----------------------------
# URL ingestion
# -----------------------------
if source_type == "URLs":
    num_links = st.sidebar.slider("How many links?", min_value=1, max_value=5, value=1)
    urls = [
        st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
        for i in range(num_links)
    ]
    urls = [u for u in urls if u.strip()]

    if st.sidebar.button("Process URLs"):
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            with st.spinner("Loading and indexing URLs..."):
                try:
                    loader = UnstructuredURLLoader(urls=urls)
                    docs = loader.load()
                    url_docs = splitter.split_documents(docs)
                    st.session_state.url_vectorstore = FAISS.from_documents(
                        url_docs, embeddings
                    )
                    st.success(f"Indexed {len(url_docs)} chunks from {len(urls)} URL(s).")
                except Exception as e:
                    st.error(f"Failed to process URLs: {e}")

# -----------------------------
# PDF ingestion
# -----------------------------
if source_type == "PDF":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        st.session_state.pdf_bytes = uploaded_file.getvalue()
        st.session_state.pdf_name = uploaded_file.name

        if st.sidebar.button("Process PDF"):
            with st.spinner("Reading and indexing PDF..."):
                try:
                    pdf_reader = PdfReader(uploaded_file)
                    pdf_text = "".join(
                        page.extract_text() or "" for page in pdf_reader.pages
                    )
                    if not pdf_text.strip():
                        st.warning("No extractable text found in this PDF (it may be scanned/image-based).")
                    else:
                        pdf_chunks = splitter.split_text(pdf_text)
                        st.session_state.pdf_vectorstore = FAISS.from_texts(
                            pdf_chunks, embeddings
                        )
                        st.success(f"Indexed {len(pdf_chunks)} chunks from {uploaded_file.name}.")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

# =============================
# Q&A interface
# =============================
st.header("Ask Questions")

if source_type == "URLs":
    query = st.text_input("Ask your question about the URLs:")
    if query:
        if st.session_state.url_vectorstore is None:
            st.warning("Please process at least one URL first.")
        else:
            with st.spinner("Thinking..."):
                rag_chain = create_retrieval_chain(
                    st.session_state.url_vectorstore.as_retriever(),
                    qa_document_chain,
                )
                result = rag_chain.invoke({"input": query})
            st.subheader("Answer")
            st.write(result["answer"])
            sources = {
                doc.metadata.get("source")
                for doc in result.get("context", [])
                if doc.metadata.get("source")
            }
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")

elif source_type == "PDF":
    query = st.text_input("Ask your question about the PDF:")
    if query:
        if st.session_state.pdf_vectorstore is None:
            st.warning("Please process a PDF first.")
        else:
            with st.spinner("Thinking..."):
                relevant_docs = st.session_state.pdf_vectorstore.similarity_search(query)
                response = qa_document_chain.invoke(
                    {"input": query, "context": relevant_docs}
                )
            st.subheader("Answer")
            st.write(response)

    # -----------------------------
    # PDF Summarization
    # -----------------------------
    if st.session_state.pdf_bytes and st.button("Summarize PDF"):
        with st.spinner("Generating summary..."):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(st.session_state.pdf_bytes)
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load_and_split()

                summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = summary_chain.invoke({"input_documents": docs})

                st.subheader("PDF Summary")
                st.write(summary["output_text"])
            except Exception as e:
                st.error(f"Failed to summarize PDF: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)