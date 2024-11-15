from langchain.document_loader import DirectoryLoader

DATA_PATH = "data/pdf"

def load_documents():
    loader = DirectoryLoader(DATA_PATH,glob="*.md")
    documents = loader.load()
    
    return documents

