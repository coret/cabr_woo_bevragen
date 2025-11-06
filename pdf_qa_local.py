from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import os

PDF_FOLDER = "pdfs"
CHROMA_DB = "chroma_db"
MODEL_NAME = "llama3"

def load_pdfs(pdf_folder):
    docs = []
    for f in os.listdir(pdf_folder):
        if f.endswith(".pdf"):
            docs.extend(PyPDFLoader(os.path.join(pdf_folder, f)).load())
    return docs

def build_vectorstore(docs):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding=emb, persist_directory=CHROMA_DB)
    db.persist()
    return db

def run_llm(llm, prompt, context, question):
    full_prompt = prompt.format(context=context, question=question)
    return llm.invoke(full_prompt)

if __name__ == "__main__":
    if not os.path.exists(CHROMA_DB):
        db = build_vectorstore(load_pdfs(PDF_FOLDER))
    else:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_DB, embedding_function=emb)

    retriever = db.as_retriever()
    llm = OllamaLLM(model=MODEL_NAME)

    template = """Gebruik de onderstaande context om de vraag te beantwoorden.
    Context: {context}
    Vraag: {question}"""
    prompt = PromptTemplate.from_template(template)

    while True:
        q = input("\nStel een vraag (of 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        docs = retriever.invoke(q)
        context = "\n\n".join(d.page_content for d in docs)
        print("\n💬", run_llm(llm, prompt, context, q))
