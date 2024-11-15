# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

DATA_PATH = "data/pdf"
CHROMA_PATH = "chroma"

def load_documents():
    
    #loader = DirectoryLoader(DATA_PATH,glob="*.md")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader) # Load PDF files
    documents = loader.load()
    
    return documents

def split_text(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)   
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[10]            # Get the 10th document
    print(document.page_content)     # Print the content of the document
    print(document.metadata)         # Print the metadata of the document

    return chunks


documents = load_documents()
split_text(documents)
