import streamlit as st
import time
import random
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import getpass
import os
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings



# Page Configuration
st.set_page_config(page_title="Chatbot UI", page_icon="💬")
st.title("💬 Simple Chatbot")

# Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_run" not in st.session_state:
    st.session_state.first_run = True

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.rerun()
    llm_api_key = st.text_input("LLM API Key", type="password", placeholder="Enter your MistralAI API key here")
    web_source_link_1 = st.text_input("First Web Source Link", placeholder="Enter the URL of the source document")
    web_source_link_2 = st.text_input("Second Web Source Link", placeholder="Enter the URL of the source document")
    web_source_link_3 = st.text_input("Third Web Source Link", placeholder="Enter the URL of the source document")
    if st.button("🔗 Add Web Source Links"):
        st.session_state.web_sources = [web_source_link_1, web_source_link_2, web_source_link_3]
        URLs = []
        for link in st.session_state.web_sources:
            if link:
                URLs.append(link)
        loader = UnstructuredURLLoader(urls=URLs)
        data = loader.load()
        
        # Embed chunks and create vector store
        # Create vector store using the loaded chunks and the HuggingFaceEmbeddings model
        chunks = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200).split_documents(data)
        # Initialize the embedding model
        # Using HuggingFaceEmbeddings directly as it's more standard with Langchain and already installed
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        # Initialize LLM
        if not os.environ.get("MISTRAL_API_KEY"):
            os.environ["MISTRAL_API_KEY"] = llm_api_key
        llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
        
        # Create retriever
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
        
        # Define a prompt template for the RAG chain
        # This template structures the input for the LLM, including context and the question.
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        """
        RAG_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create RAG chain using RetrievalQA with the defined prompt
        # This is a common and straightforward way to build a RAG chain with Langchain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # 'stuff' chain type puts all retrieved documents into the prompt
            retriever=retriever,
            return_source_documents=True, # Keep this to see the source chunks
            chain_type_kwargs={"prompt": RAG_PROMPT} # Pass the custom prompt here
        )
        st.session_state.rag_chain = rag_chain
        st.session_state.messages.append({"role": "assistant", "content": "Web source links added successfully!"})
        # Display success message
        st.success("Web source links added successfully!")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Simulated Bot Response (Replace with your logic)
def get_bot_response(user_input):
    
    result = rag_chain.invoke({"query": user_input}) # Use 'query' as the input key for RetrievalQA chain  
    # Simulate processing time
    time.sleep(0.5)
    
    # Return random response
    return result['result']

# Chat Interface
if prompt := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Simulate typing animation
        bot_response = get_bot_response(prompt)
        for chunk in bot_response.split():
            full_response += chunk + " "
            time.sleep(0.1)
            response_placeholder.markdown(full_response + "▌")
        
        # Final response
        response_placeholder.markdown(full_response)
    
    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})