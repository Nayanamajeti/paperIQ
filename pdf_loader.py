from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    reader = PdfReader(str(path))
    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "page": page_number},
            )
        )

    return documents
