import os
import shutil
import json
import tkinter as tk
from tkinter import filedialog
from langchain_unstructured import UnstructuredLoader #to read various file types
from datetime import datetime

from lib import initialize_components, extract_rfp_outline, generate_answer_for_question #our custom library

def main():
    print("--- RFP Processing Workflow Initialized ---")
    
    # --- STEP 1: LOAD RFP DOCUMENT ---
    upload_folder = "uploaded_rfps"
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)
    os.makedirs(upload_folder, exist_ok=True)

    # Use Tkinter to open a file selection dialog
    print("Opening file browser to select an RFP document...")
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an RFP Document",
        filetypes=(
            ("PDF Documents", "*.pdf"),
            ("Word Documents", "*.docx"),
            ("Text Files", "*.txt"),
            # ("All files", "*.*")
        )
    )
    if not file_path:
        print("No file selected. Workflow cancelled.")
        shutil.rmtree(upload_folder)
        return
    try:
        destination_path = os.path.join(upload_folder, os.path.basename(file_path))
        shutil.copy(file_path, destination_path)
        print(f"Successfully copied '{os.path.basename(file_path)}' to '{upload_folder}' folder.")
    except Exception as e:
        print(f"Error copying file: {e}")
        return

    # --- INITIALIZE AI COMPONENTS ---
    llm, rag_chain = initialize_components()

    # --- STEP 2: RUN OUTLINER AGENT ---
    outline = extract_rfp_outline(destination_path, llm)
    if not outline:
        print("Workflow terminated.")
        return

    # --- STEP 3: USER APPROVAL (HUMAN-IN-THE-LOOP) ---
    print("\n--- Extracted RFP Outline ---")
    print(json.dumps(outline, indent=2))
    print("----------------------------")

    # Create a folder to store the outlines if it doesn't exist
    try:
        outline_folder = "generated_outlines"
        os.makedirs(outline_folder, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(destination_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = os.path.join(outline_folder, f"{base_filename}_outline_{timestamp}.json")

        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(outline, f, indent=2)
        print(f"[Info] Outline successfully saved to: {json_filename}")
    except Exception as e:
        print(f"[Error] Error saving JSON file: {e}")
    # -- end save outline --
    
    approval = input("Does this outline look correct? (yes/no): ").lower()
    if approval != 'yes':
        print("Workflow aborted by user.")
        return

    # --- STEP 4: RUN RESPONDER AGENT & FINAL ASSEMBLY ---
    print("\n--- Running Responder Agent on Approved Outline ---")
    final_response_doc = ""
    for section in outline:
        section_title = section.get("section_title", "Untitled Section")
        final_response_doc += f"## {section_title}\n\n"
        
        for question in section.get("questions", []):
            print(f"Answering: '{question}'...")
            answer = generate_answer_for_question(question, rag_chain)
            final_response_doc += f"### Question:\n{question}\n\n"
            final_response_doc += f"### Answer:\n{answer}\n\n"
            final_response_doc += "-----------------------------------\n\n"

    # --- STEP 5: SAVE FINAL DOCUMENT ---
    output_filename = "RFP_Response.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_response_doc)
        
    print(f"\n[Completed] Success! The complete response has been saved to '{output_filename}'")

if __name__ == "__main__":
    main()