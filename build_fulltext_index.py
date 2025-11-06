from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser
import os
from langchain_community.document_loaders import PyPDFLoader

PDF_FOLDER = "pdfs"
INDEX_DIR = "whoosh_index"

schema = Schema(
    path=ID(stored=True, unique=True),
    content=TEXT(stored=True)
)

if not os.path.exists(INDEX_DIR):
    os.mkdir(INDEX_DIR)
    ix = create_in(INDEX_DIR, schema)
    writer = ix.writer()

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            docs = loader.load()
            text = "\n".join(d.page_content for d in docs)
            writer.add_document(path=file, content=text)
            print(f"📄 Geïndexeerd: {file}")
    writer.commit()
else:
    print("✅ Bestaande index gevonden.")
