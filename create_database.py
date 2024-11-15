import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

# Prompt the user for the API key
OPENAI_API_KEY = input("Please enter your OpenAI API key: ")

# Use the API key in your code
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_PATH = "data/pdf"
DATA_PATH_articles = "data/articles"
CHROMA_PATH = "chroma"
CHROMA_PATH_articles = "chroma_articles"

def main():
    documents = load_documents(data_path=DATA_PATH)
    documents_scientific = load_documents(data_path=DATA_PATH_articles)
    chunks = split_text(documents)
    chunks_scientific = split_text(documents_scientific)
    save_to_chroma(chunks, CHROMA_PATH)
    save_to_chroma(chunks_scientific, CHROMA_PATH_articles)

def load_documents(data_path: str):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = []
    for doc in loader.lazy_load():
        try:
            documents.append(doc)
        except Exception as e:
            print(f"Error loading file {doc}: {e}")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[10]  # Get the 10th document
    print(document.page_content)  # Print the content of the document
    print(document.metadata)  # Print the metadata of the document

    return chunks

def save_to_chroma(chunks: list[Document], chroma_path: str):
    # Clear out the database first
    if os.path.exists(chroma_path):
        for _ in range(5):  # Retry up to 5 times
            try:
                shutil.rmtree(chroma_path)
                break
            except PermissionError:
                print("PermissionError: Retrying in 1 second...")
                time.sleep(1)
        else:
            print("Failed to delete the directory after 5 attempts.")
            return

    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)  # persist_directory is the directory where the database is saved
    db.persist()  # Save the database
    print(f"Saved {len(chunks)} chunks to {chroma_path}")

if __name__ == "__main__":
    main()