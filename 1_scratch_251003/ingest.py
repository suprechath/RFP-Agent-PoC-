import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores.utils import filter_complex_metadata

KNOWLEDGE_BASE_DIR = "../helper/knowledge_base"
DB_DIR = "db"

print("Starting ingestion process...")

# Load documents from the knowledge_base folder
loader = PyPDFDirectoryLoader(KNOWLEDGE_BASE_DIR)
docs = loader.load()
if not docs:
    print("No documents found in the knowledge_base directory. Please add some PDFs.")
    exit()

print(f"Loaded {len(docs)} page(s).")

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Split documents into {len(splits)} chunks.")

# Create embeddings model
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store and save it to disk
print("Creating and persisting vector store...")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory=DB_DIR
)

print(f"Ingestion complete! Vector store created and saved to '{DB_DIR}' directory.")
