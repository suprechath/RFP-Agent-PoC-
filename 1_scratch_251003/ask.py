import sys
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DB_DIR = "db"

# Load the existing vector store
print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant chunks

# Set up the LLM
print("Loading LLM...")
llm = Ollama(model="llama3.1:8b")

# Create the prompt template
prompt_template = """
You are a helpful assistant for our sales team. 
Answer the teammate's request for building the perfect proposal.
If the context doesn't contain the answer, say that you don't have enough information.

Context:
{context}

Question:
{input}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

# Create the chain that combines the prompt, LLM, and retrieved documents
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\nReady to answer your questions! Type 'exit' to quit.")

# Create an interactive loop
while True:
    question = input("\nYou: ")
    if question.lower() == 'exit':
        print("Exiting...")
        break
    
    # Invoke the chain with the user's question
    response = rag_chain.invoke({"input": question})
    
    # Print the answer in a clean way
    print(f"\nAI Assistant: {response['answer']}")