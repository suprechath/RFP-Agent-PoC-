# --- Model and Embedding Settings ---
MODEL_NAME = "llama3:8b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Directory Settings ---
DB_DIR = "db"

# --- Prompt Templates ---

OUTLINER_PROMPT_TEMPLATE = """
You are an expert RFP analyst. Your task is to meticulously parse the following document text and extract all questions and required sections. 
Structure your output as a clean, valid JSON object. The JSON should be a list of dictionaries, where each dictionary has three keys: "section_number", "section_title", and a "questions" list.
Do not include any text or explanations outside of the JSON object itself.

---
Document Text: 
{document_text}
---

JSON Output:
"""

RESPONDER_PROMPT_TEMPLATE = """
You are a professional proposal writer for a company named 'Innovate Inc.'. 
Your task is to write a clear, concise, and professional answer to the client's question using ONLY the provided context.
If the context does not contain the information needed to answer the question, you must state: "Based on the provided documents, I do not have enough information to answer this question."

---
Context:
{context}
---
Client Question:
{input}
---

Answer:
"""