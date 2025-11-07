import os
import warnings
import logging

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Phase 2 libraries - Updated for OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 libraries - UPDATED IMPORTS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('RAG Chatbot! Ask any thing from your documents 📄 🤖')

# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Phase 3 (Pre-requisite) - UPDATED VECTORSTORE CREATION
@st.cache_resource
def get_vectorstore():
    pdf_name = "Abdullah kaimkhani CV.pdf"
    
    # Load the PDF
    loader = PyPDFLoader(pdf_name)
    documents = loader.load()
    
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

# Function to format documents for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG system prompt template
prompt_template = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
                                        the most accurate and most precise answers. 
                                        
                                        Use the following context to answer the question. If the context doesn't contain 
                                        the answer, say you don't know based on the provided information.
                                        
                                        Context: {context}
                                        
                                        Question: {question}
                                        
                                        Start the answer directly. No small talk please.""")

# Initialize OpenAI chat model
def get_openai_chat():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.error("OpenAI API key not found. Please check your .env file")
        return None
        
    return ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",  # You can change to "gpt-4" if you have access
        temperature=0.1
    )

prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role':'user', 'content': prompt})
    
    try:
        # Initialize OpenAI chat
        openai_chat = get_openai_chat()
        
        if openai_chat is None:
            st.stop()
        
        # Phase 3 - RAG with OpenAI
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document")
            st.stop()
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        
        # Create RAG chain using the newer syntax
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | openai_chat
            | StrOutputParser()
        )
        
        # Get response
        response = rag_chain.invoke(prompt)
        
        # Display response
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append(
            {'role':'assistant', 'content': response})
            
    except Exception as e:
        st.error(f"Error: {str(e)}")