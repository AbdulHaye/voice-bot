from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma

from dotenv import load_dotenv
import os
import shutil

load_dotenv()

CHROMA_PATH = "chroma"
PDF_PATH = "data/script.pdf"

from log_setup import setup_logger

logger = setup_logger()


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[0]
    logger.info(document.page_content)
    logger.info(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    Chroma.from_documents(
        chunks,
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", task_type="semantic_similarity"
        ),
        persist_directory=CHROMA_PATH,
    )
    logger.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
