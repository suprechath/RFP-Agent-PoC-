import os
import json
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain_community.document_loaders import UnstructuredFileLoader //deprecated
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# Import settings from our config file
from config import MODEL_NAME, EMBEDDING_MODEL_NAME, DB_DIR, OUTLINER_PROMPT_TEMPLATE, RESPONDER_PROMPT_TEMPLATE

def initialize_components():
    """Initializes and returns the LLM, retriever, and responder chain."""
    llm = Ollama(model=MODEL_NAME)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks

    responder_prompt = PromptTemplate.from_template(RESPONDER_PROMPT_TEMPLATE)
    question_answer_chain = create_stuff_documents_chain(llm, responder_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return llm, rag_chain

def extract_rfp_outline(rfp_filepath, llm):
    """
    Agent 1: The Outliner.
    Reads the RFP document and asks the LLM to create a JSON outline.
    """
    print(f"--- Running Outliner Agent on {rfp_filepath} ---")

    # Get the file extension to decide which loader to use
    file_extension = os.path.splitext(rfp_filepath)[1].lower()
    print(f"Detected file extension: {file_extension}")
    docs = []
    try:
        if file_extension == ".pdf":
            print("Using PyPDFLoader for PDF file...")
            loader = PyPDFLoader(rfp_filepath)
            docs = loader.load_and_split()
        elif file_extension == ".docx":
            print("Using Docx2txtLoader for Word file...")
            loader = Docx2txtLoader(rfp_filepath)
            docs = loader.load()
        elif file_extension == ".txt":
            print("Using built-in loader for text file...")
            with open(rfp_filepath, "r", encoding="utf-8") as f:
                text_content = f.read()
            docs = [Document(page_content=text_content)]
        else:
            print(f"Error: Unsupported file type '{file_extension}'. Please use PDF, DOCX, or TXT.")
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    # Combine the content from all pages/documents into a single string
    document_text = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
    
    if not document_text:
        print("Error: Could not read document text.")
        return None

    # Create the prompt
    outliner_prompt = PromptTemplate.from_template(OUTLINER_PROMPT_TEMPLATE)
    chain = outliner_prompt | llm
    
    # Invoke the chain to get the JSON outline
    response_text = chain.invoke({"document_text": document_text})
    
    # A crucial step: try to parse the LLM's JSON output
    try:
        # Sometimes the LLM might add markdown ```json ... ```, so we clean it
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        
        outline = json.loads(response_text)
        print("--- Outline extracted successfully! ---")
        return outline
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON output from the LLM.")
        print("LLM Raw Output:\n", response_text)
        return None

def generate_answer_for_question(question, rag_chain):
    """
    Agent 2: The Responder.
    Takes a single question and uses the RAG chain to generate an answer.
    """
    response = rag_chain.invoke({"input": question})
    return response['answer']