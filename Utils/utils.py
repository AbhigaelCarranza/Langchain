from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import pickle
import os
import streamlit as st
from dotenv import load_dotenv


def load_documents():
    file="/mnt/c/Users/amendez/github/Langchain/Files/ARINC_653P1-2-Div.pdf"
    pdf=PyPDFLoader(file)
    loaders=pdf.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120,length_function=len)
    return text_splitter.split_documents(loaders)

def load_documents_from_web():
    docs = []
    uploaded_files = st.file_uploader("Upload PDF: ",type=['pdf'])
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load()) 
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200,length_function=len)
    return text_splitter.split_documents(docs)

def get_faiss_vectorStore(embeddings):
    if os.path.exists("arinc.pkl"):
        with open("arinc.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        chunks=load_documents()
        vector_store=FAISS.from_documents(chunks, embeddings)
        with open("arinc.pkl", "wb") as f:
            pickle.dump(vector_store, f)
    return vector_store