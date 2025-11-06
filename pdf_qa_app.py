import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser

PDF_FOLDER = "pdfs"
CHROMA_DB = "chroma_db"
WHOOSH_INDEX = "whoosh_index"
MODEL_NAME = "llama3"

# =====================================================
# INDEXERING
# =====================================================
@st.cache_resource(show_spinner=False)
def build_fulltext_index():
    """Maak of laad de Whoosh-index met PDF-teksten."""
    schema = Schema(path=ID(stored=True, unique=True), content=TEXT(stored=True))
    if not os.path.exists(WHOOSH_INDEX):
        os.mkdir(WHOOSH_INDEX)
        ix = create_in(WHOOSH_INDEX, schema)
        writer = ix.writer()
        for f in os.listdir(PDF_FOLDER):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, f))
                docs = loader.load()
                text = "\n".join(d.page_content for d in docs)
                writer.add_document(path=f, content=text)
        writer.commit()
    return open_dir(WHOOSH_INDEX)

def search_fulltext(query, limit=10):
    ix = build_fulltext_index()
    parser = MultifieldParser(["content"], schema=ix.schema)
    q = parser.parse(query)
    results_list = []
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=limit)
        for r in results:
            snippet = r.highlights("content") or r["content"][:400] + "..."
            results_list.append((r["path"], snippet))
    return results_list

# =====================================================
# VECTOR / LLM
# =====================================================
@st.cache_resource(show_spinner=False)
def load_vectorstore():
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

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="PDF Chat & Zoeker", page_icon="💬", layout="wide")
st.title("💬 Chat en 🔍 Zoek in je PDF-documenten (lokaal via Ollama)")

with st.spinner("📦 Laden van index en vectorstore..."):
    db = load_vectorstore()
    retriever = db.as_retriever()
    ix = build_fulltext_index()

llm = OllamaLLM(model=MODEL_NAME)
template = """Beantwoord de vraag in het Nederlands op basis van de onderstaande context.
Context:
{context}

Vraag:
{question}"""
prompt = PromptTemplate.from_template(template)

# Chatgeschiedenis
if "messages" not in st.session_state:
    st.session_state.messages = []

# Toon eerdere berichten
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Gebruikte context"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** (pagina {src['page']})")
                    st.write(src["snippet"])
                    st.divider()

# Invoer
if question := st.chat_input("Stel een vraag of typ /zoek <term> voor full-text zoeken..."):
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # -------- Full-text zoekmodus --------
    if question.lower().startswith("/zoek "):
        term = question.split("/zoek ", 1)[1]
        with st.spinner(f"🔍 Zoeken naar '{term}'..."):
            results = search_fulltext(term)
        with st.chat_message("assistant"):
            if not results:
                st.write("Geen resultaten gevonden.")
            else:
                for path, snippet in results:
                    st.markdown(f"**📄 {path}**")
                    st.write(snippet)
                    st.divider()
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"🔍 Resultaten voor '{term}'",
            "sources": [{"source": path, "page": "", "snippet": snip} for path, snip in results]
        })

    # -------- AI / semantische modus --------
    else:
        with st.spinner("🧠 Analyseren met Llama3..."):
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            answer = run_llm(llm, prompt, context, question)
        sources = []
        for d in docs:
            meta = d.metadata or {}
            snippet = d.page_content[:400].replace("\n", " ") + "..."
            sources.append({
                "source": meta.get("source", "Onbekend document"),
                "page": meta.get("page", "n.v.t."),
                "snippet": snippet
            })
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("📄 Gebruikte context"):
                for src in sources:
                    st.markdown(f"**{src['source']}** (pagina {src['page']})")
                    st.write(src["snippet"])
                    st.divider()
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
