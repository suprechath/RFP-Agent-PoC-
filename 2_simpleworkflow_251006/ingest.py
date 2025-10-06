import os
import shutil
from langchain_unstructured import UnstructuredLoader #to read various file types
from langchain_community.vectorstores.utils import filter_complex_metadata #to filter out complex metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter #to split large documents
from langchain_huggingface import HuggingFaceEmbeddings #to create embeddings
from langchain_community.vectorstores import Chroma #to create and manage the vector store

# Import settings from our config file
from config import KNOWLEDGE_BASE_DIR, DB_DIR, EMBEDDING_MODEL_NAME

 # Main function to handle the ingestion of documents into the vector store.
def main():
    # Check if a database already exists and ask for confirmation to overwrite
    if os.path.exists(DB_DIR):
        print(f"Database directory '{DB_DIR}' already exists.")
        overwrite = input("Do you want to overwrite it? (yes/no): ").lower()
        if overwrite == 'yes':
            print("Removing existing database...")
            shutil.rmtree(DB_DIR)
        else:
            print("Ingestion cancelled by user.")
            return

    # Check if the knowledge base directory exists
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Error: Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
        print("Please create it and add your documents.")
        return

    # Get a list of all files to process
    all_files = [
        f for f in os.listdir(KNOWLEDGE_BASE_DIR) 
            if os.path.isfile(os.path.join(KNOWLEDGE_BASE_DIR, f))
    ]
    if not all_files:
        print(f"No documents found in '{KNOWLEDGE_BASE_DIR}'. Please add your files.")
        return
    print(f"[Info] Found {len(all_files)} documents to process in '{KNOWLEDGE_BASE_DIR}'.")
    
    # Load all documents from the directory
    all_docs = []
    for file_name in all_files:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)
        try:
            print(f"Loading document: {file_path}")
            loader = UnstructuredLoader(file_path)
            all_docs.extend(loader.load()) 
            print(f"  - [Successful] loaded {file_name}")
        except Exception as e:
            print(f"  - [Failed] to load {file_name}: {e}")   
    if not all_docs:
        print("Could not load any documents. Aborting ingestion.")
        return
    print(f"[Info] Total documents loaded: {len(all_docs)}")

    # Function to filter out documents with complex metadata
    print("[Info] Filtering complex metadata from documents...")
    filtered_docs = filter_complex_metadata(all_docs)


    # Split the filtered_docs into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(filtered_docs)
    print(f"[Info] Split documents into {len(splits)} chunks.")

    # Create the embeddings model
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create the vector store and persist it
    print(f"[Info] Creating and saving vector store to '{DB_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"\n[Completed] Ingestion complete! Your knowledge base is ready.")

if __name__ == "__main__":
    main()