"""Ingest a PDF into Qdrant (embed + store)."""

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

PDF_PATH = "documents/fxhive.pdf"
COLLECTION = "fxhive_documents"
QDRANT_URL = "http://localhost:6333"


def main() -> None:
    # 1) Load PDF pages
    docs = PyPDFLoader(PDF_PATH).load()
    print(f"Pages loaded: {len(docs)}")

    # 2) Split into overlapping chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    ).split_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    # 3) Recreate collection so re-ingest stays clean
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    # 4) Embed + store in Qdrant
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )
    print("PDF indexed successfully!")


if __name__ == "__main__":
    main()
