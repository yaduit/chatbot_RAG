import streamlit as st
from pypdf import PdfReader
import pandas as pd
import base64

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import promptTemplate

from datetime import datetime

#To get text chunks from pdf

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf(header) = PdfReader(pdf)
        for page in PdfReader.pages:
            text += page.extract_text()
            return text
        

#To get chunks from text

def get_text_chunks(text,model_name):
    if(model_name) == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Embedding this chunks and storing them in a vector store

def get_vector_store(text_chunks, model_name, api_key=None):
    if(model_name) == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001", google_api_key = api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

#create a conversational chain using langchain

