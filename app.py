import asyncio
import os
import logging

# Ensure asyncio event loop is initialized properly
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set OpenMP environment variable to avoid conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Disable Streamlit watchdog logger to avoid conflicts with PyTorch
logging.getLogger('watchdog').setLevel(logging.ERROR)

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemma model and tokenizer
@st.cache_resource
def load_gemma():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Load the model once at startup
gemma_pipeline = load_gemma()

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from images using OCR
def get_image_text(images):
    text = ""
    for img in images:
        pil_img = Image.open(img)
        text += pytesseract.image_to_string(pil_img)
    return text

# Function to extract text from a URL
def get_url_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    texts = soup.stripped_strings
    return " ".join(texts)

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create FAISS vector store
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load FAISS index and perform retrieval
def retrieve_context(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# Function to generate response using Gemma
def generate_response(question, context):
    prompt_template = """
    Given the following context, answer the question in a concise manner:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = prompt_template.format(context=context, question=question)
    
    # Generate text using Gemma model
    result = gemma_pipeline(prompt)[0]['generated_text']
    
    final_answer = result.split("Answer:")[-1].strip() 

    return final_answer


# Function to handle file processing
def process_input(file_option, file_data=None, url_data=None):
    if file_option == "PDF":
        if file_data:
            raw_text = get_pdf_text(file_data)
        else:
            st.error("Please upload at least one PDF file.")
            return None
    elif file_option == "Image":
        if file_data:
            raw_text = get_image_text(file_data)
        else:
            st.error("Please upload at least one image.")
            return None
    elif file_option == "URL":
        if url_data:
            raw_text = get_url_text(url_data)
        else:
            st.error("Please provide a URL.")
            return None
    else:
        st.error("Invalid input type.")
        return None

    # Process and store the text
    chunks = get_text_chunks(raw_text)
    create_vector_store(chunks)
    return raw_text

# Streamlit UI
def main():
    st.title("Gemma-3-1B Powered RAG System for Multi-Source Document Retrieval and Question Answering")
    # Sidebar for input type selection
    with st.sidebar:
        st.header("Upload Files or Provide URL")
        
        # Options for the user
        file_option = st.radio("Choose the type of input:", ("PDF", "Image", "URL"))
        
        file_data = None
        url_data = None
        
        if file_option == "PDF":
            file_data = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        elif file_option == "Image":
            file_data = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        elif file_option == "URL":
            url_data = st.text_input("Enter URL to process")

        if st.button(f"Process {file_option}s"):
            with st.spinner("Processing..."):
                raw_text = process_input(file_option, file_data, url_data)
                if raw_text:
                    st.success(f"{file_option} processed and stored in FAISS!")

    # User input for question
    question = st.text_input("Ask a question about the uploaded content:")
    
    if question:
        with st.spinner("Retrieving and generating response..."):
            context = retrieve_context(question)
            if context:
                response = generate_response(question, context)
                st.write("**Answer:**")
                st.write(response)
            else:
                st.write("No relevant information found in the document.")

if __name__ == "__main__":
    main()