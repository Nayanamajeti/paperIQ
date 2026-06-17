import os

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import get_embeddings
from pdf_loader import load_pdf


def load_and_index_pdfs(pdf_folder="data/"):
    documents = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            documents.extend(load_pdf(os.path.join(pdf_folder, filename)))

    if not documents:
        print("No PDFs found in data/ folder!")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")

    embeddings = get_embeddings()
    print("Building FAISS index (this may take a few minutes for large PDFs)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index saved!")
    return vectorstore


if __name__ == "__main__":
    load_and_index_pdfs()
