import streamlit as st
import time
import os


# LangChain imports
#from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
# --- Streamlit Page Setup ---
from llms.gemini_model_ import get_gemini
from llms.mistralai import get_mistralai_model

st.set_page_config(
    page_title="Custom Web Chatbot", page_icon="🤖")
st.title("🤖 Custom Web Chatbot")

# --- Session State Defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_chain" not in st.session_state:
    st.session_state.conv_chain = None

# --- Sidebar Controls -----
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    load_dotenv()
    gemini_api_key = os.getenv("GEMINIAPI_KEY")
    mistralai_api_key = os.getenv("MISTRALAI_KEY")


    
    urls = [] #load the URLs
    for i in range(1, 4):
        link = st.text_input(
            f"Web Source Link {i}", placeholder="https://example.com/..."
        )
        if link:
            urls.append(link)

    if st.button("🔗 Load & Index Sources"):
        if not urls:
            st.error("❌ Please enter at least one URL.")
        else:
            try:
                # 1) Load documents
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                if not data:
                    raise ValueError(
                        "No content loaded from the provided URLs.")

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

                # 4) LLM setup with fallback
                llm = None
                # Try MistralAI
                try:

                    if "GEMINIAPI_KEY" in os.environ:

                        llm = get_gemini()
                        _ = llm.invoke("ping")
                        st.success("✅ Using Gemini as primary LLM.")
                    else:
                        raise ValueError(
                            "No Gemini key provided .")
                except Exception as gemini_err:
                    st.error(f"⚠️ Gemini failed {gemini_err}")
                    # Fallback to Gemini
                    try:
                        if "MISTRALAI_KEY" in os.environ:

                            llm = get_mistralai_model()
                            # Test connection
                            _ = llm.invoke("ping")
                            st.success("✅ Using MistralAI as fallback LLM.")
                        else:
                            raise ValueError(
                                "No MistralAI key provided for fallback.")
                    except Exception as mistral_err:

                        st.warning(f"❌ Both LLMs failed: {mistral_err}")
                    raise gemini_err

                # 5) Retriever setup
                retriever = vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 3})

                # 6) Prompt template
                prompt_template = '''
                                    You are a friendly, talkative assistant. Use the following pieces of context to answer the question at the end.
                                    If you don't know the answer, just say that you don't know.

                                    {context}

                                    Question: {question}
                                
                                   '''
                RAG_PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"],
                )

                # 7) Memory
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                )

                # 8) Conversational RAG chain
                conv_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
                )
                #Initiates conversation
                st.session_state.messages = [{"role": "assistant", "content": "Hi, I am Zeema! How may I help you today?🙂"}]
                st.session_state.conv_chain = conv_chain
                st.success("🚀 Sources loaded and conversation chain ready!")

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
        bot_reply = "❌ Please load sources first in the sidebar."

    # Simulate typing effect
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for word in bot_reply.split():
            full_text += word + " "
            time.sleep(0.05)
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

    # Log assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_text})



