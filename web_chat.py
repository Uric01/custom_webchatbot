import streamlit as st
import time
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Page Configuration
st.set_page_config(page_title="Custom Web Chatbot", page_icon="💬")
st.title("💬 Custom Web Chatbot")

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
        try:
            if not llm_api_key:
                st.error("❌ Please provide your MistralAI API key.")
                st.stop()

            urls = list(filter(None, [web_source_link_1, web_source_link_2, web_source_link_3]))
            if not urls:
                st.error("❌ Please provide at least one valid URL.")
                st.stop()

            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            if not data:
                st.error("❌ Failed to load content from the provided URLs.")
                st.stop()

            chunks = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200).split_documents(data)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)

            os.environ["MISTRAL_API_KEY"] = llm_api_key
            llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
            prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
            RAG_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": RAG_PROMPT}
            )
            st.session_state.rag_chain = rag_chain
            st.session_state.messages.append({"role": "assistant", "content": "Web source links added successfully!"})
            st.success("✅ Web source links added successfully!")

        except Exception as e:
            st.error(f"❌ Error loading sources or initializing chatbot: {str(e)}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define bot response function with error handling
def get_bot_response(user_input, rag_chain):
    try:
        result = rag_chain.invoke({"query": user_input})
        time.sleep(0.5)
        return result["result"]
    except Exception as e:
        return f"❌ Failed to get response from model: {str(e)}"

# Chat Interface
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        if "rag_chain" in st.session_state:
            bot_response = get_bot_response(prompt, st.session_state.rag_chain)
        else:
            bot_response = "❌ RAG chain is not initialized. Please add web source links first."

        for chunk in bot_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
