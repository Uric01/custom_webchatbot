
# 🤖 Custom Web Chatbot

A conversational web-based chatbot built with **Streamlit** and powered by **LangChain**, **Gemini**, and **MistralAI**. This app lets you ingest up to 3 web URLs, indexes their content using embeddings, and enables users to ask context-aware questions based on that information. It supports conversation memory and fallback to alternate LLMs if needed.

---

## 🚀 Features

- 🧠 Conversational RAG using Gemini or MistralAI
- 🌐 Loads and parses up to 3 web pages using `UnstructuredURLLoader`
- 📚 Splits text into manageable chunks using `CharacterTextSplitter`
- 🧬 Embeds text with HuggingFace's `all-mpnet-base-v2` model
- ⚡ Retrieves relevant information with FAISS similarity search
- 🔁 Supports persistent chat history using Streamlit session state
- 🛠️ Sidebar for source input, conversation reset, and LLM setup
- 💬 Simulated typing effect for assistant responses

---

## 🧱 Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Hugging Face Sentence Transformers](https://www.sbert.net/)
- [Gemini API (Google)](https://ai.google.dev/)
- [MistralAI](https://mistral.ai/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Python-dotenv](https://pypi.org/project/python-dotenv/)

---

## ⚙️ Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/your-username/custom-web-chatbot.git
cd custom-web-chatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Create `.env` file**

```env
GEMINIAPI_KEY=your_gemini_api_key
MISTRALAI_KEY=your_mistralai_api_key
```

4. **Run the app**

```bash
streamlit run your_script.py
```

---

## 🧪 How It Works

1. Enter up to 3 URLs in the sidebar.
2. Click **"Load & Index Sources"**.
3. Ask questions based on the indexed content in the chat input.
4. The chatbot uses Gemini (or MistralAI fallback) to generate answers based on similarity search from the FAISS vector store.
5. Conversation memory ensures contextual follow-ups.

---

## ❗ Notes

- Gemini and Mistral keys must be set in the `.env` file for full functionality.
- App uses `sentence-transformers/all-mpnet-base-v2` for generating embeddings.
- Fallback logic ensures the app tries Mistral if Gemini fails.

---

## 📸 Screenshot

![Chatbot Screenshot](screenshot.png)

---

## 📄 License

MIT License — feel free to fork and build on it.
