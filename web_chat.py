import streamlit as st
import time
import os

# LangChain imports
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Custom Web Chatbot", page_icon="💬")
st.title("💬 Custom Web Chatbot")

# --- Session State Defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_chain" not in st.session_state:
    st.session_state.conv_chain = None

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    llm_api_key = st.text_input(
        "LLM API Key",
        type="password",
        placeholder="Enter your MistralAI API key here",
    )
    urls = []
    for i in range(1, 4):
        link = st.text_input(f"Web Source Link {i}", placeholder="https://example.com/…")
        if link:
            urls.append(link)

    if st.button("🔗 Load & Index Sources"):
        if not llm_api_key:
            st.error("❌ Please enter your MistralAI API key.")
        elif not urls:
            st.error("❌ Please enter at least one URL.")
        else:
            try:
                # 1) Load documents
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                if not data:
                    raise ValueError("No content loaded from the provided URLs.")

                # 2) Split into chunks
                splitter = CharacterTextSplitter(
                    separator="\n", chunk_size=1000, chunk_overlap=200
                )
                chunks = splitter.split_documents(data)

                # 3) Embed & build FAISS index
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )
                vector_store = FAISS.from_documents(chunks, embeddings)

                # 4) LLM setup
                os.environ["MISTRAL_API_KEY"] = llm_api_key
                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

                # 5) Retriever
                retriever = vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 3}
                )

                # 6) Prompt template
                prompt_template = """
You are a friendly, talkative assistant. Your naname is Zeema. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know.

{context}

Question: {question}
"""
                RAG_PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )

                # 7) Memory with explicit output_key
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",  # ← fix for multiple output keys
                )

                # 8) Conversational RAG chain
                conv_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
                )

                st.session_state.conv_chain = conv_chain
                st.success("✅ Sources loaded and conversational chain initialized!")
            except Exception as e:
                st.error(f"Error setting up chain: {e}")

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Bot Response Function ---
def get_bot_response(question, conv_chain):
    try:
        result = conv_chain({"question": question})
        return result["answer"]
    except Exception as e:
        return f"❌ Model error: {e}"

# --- Chat Input & Rendering ---
if user_input := st.chat_input("Type your message here..."):
    # Log user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate bot response
    if st.session_state.conv_chain:
        bot_reply = get_bot_response(user_input, st.session_state.conv_chain)
    else:
        bot_reply = "❌ Please load web sources first in the sidebar."

    # Simulate typing
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for word in bot_reply.split():
            full_text += word + " "
            time.sleep(0.05)
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

    # Log assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_text})
