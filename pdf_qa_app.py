import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

PDF_FOLDER = "pdfs"
CHROMA_DB = "chroma_db"
MODEL_NAME = "llama3"

# ===============================
# Helpers
# ===============================
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Laad of bouw de Chroma vectorstore."""
    if not os.path.exists(CHROMA_DB):
        docs = []
        for f in os.listdir(PDF_FOLDER):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, f))
                docs.extend(loader.load())
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(docs, embedding=emb, persist_directory=CHROMA_DB)
        db.persist()
        return db
    else:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory=CHROMA_DB, embedding_function=emb)


def run_llm(llm, prompt, context, question):
    full_prompt = prompt.format(context=context, question=question)
    return llm.invoke(full_prompt)


# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="PDF Chat", page_icon="💬", layout="wide")
st.title("Chat met je PDF-documenten (lokaal via Ollama)")

with st.spinner("Laden van vector database..."):
    db = load_vectorstore()

retriever = db.as_retriever()
llm = OllamaLLM(model=MODEL_NAME)

template = """Gebruik de onderstaande context om de vraag te beantwoorden.
Context:
{context}

Vraag:
{question}"""
prompt = PromptTemplate.from_template(template)

# Initialiseer chatgeschiedenis
if "messages" not in st.session_state:
    st.session_state.messages = []

# Toon bestaande berichten
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Invoerveld onderaan
if question := st.chat_input("Stel een vraag over je documenten..."):
    # Toon vraag direct in de chat
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Ophalen context en LLM-run
    with st.spinner("Analyseren..."):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        answer = run_llm(llm, prompt, context, question)

    # Toon antwoord in de chat
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
