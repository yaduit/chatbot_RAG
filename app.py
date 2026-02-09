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
def get_conversational_chain(model_name, vector_store = None, api_key = None):
    if(model_name == "Google Ai"):
        promptTemplate = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details with proper structure , if the answer is not in the provided context just say, "answer is not available in the context", dont provide the wrong answer \n\n
            context:\n {context}?\n
            question:\n {question}?\n

            answer:
            """
        model = ChatGoogleGenerativeAI(model = "models\gemini-2.5-flash", temprature = 0.3, google_api_key = api_key)
        prompt = promptTemplate(template=prompt_template , input_variables=["context","question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    
#Take user input

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if(api_key is None or pdf_docs is None):
        st.warning("please upload any pdf and provide API key")
        return
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chunks, model_name, api_key)
    user_question_output = ""
    response_output = ""
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001", google_api_key = api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain("Google AI", vector_store=new_db, api_key=api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs = True)
        user_question_output = user_question
        response_output = response["output text"]
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),",".join(pdf_names)))


