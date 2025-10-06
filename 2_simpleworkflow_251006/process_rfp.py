import os
import json
from lib import initialize_components, extract_rfp_outline, generate_answer_for_question

def main():
    print("--- RFP Processing Workflow Initialized ---")
    
    # --- STEP 1: LOAD RFP DOCUMENT ---
    rfp_path = input("Please enter the path to the RFP document (e.g., sample_rfp.txt): ")
    if not os.path.exists(rfp_path):
        print("Error: File not found. Please check the path and try again.")
        return

    # --- INITIALIZE AI COMPONENTS ---
    llm, rag_chain = initialize_components()

    # --- STEP 2: RUN OUTLINER AGENT ---
    outline = extract_rfp_outline(rfp_path, llm)
    if not outline:
        print("Workflow terminated.")
        return

    # --- STEP 3: USER APPROVAL (HUMAN-IN-THE-LOOP) ---
    print("\n--- Extracted RFP Outline ---")
    print(json.dumps(outline, indent=2))
    print("----------------------------")
    
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
            final_response_doc += "---\n\n"

    # --- STEP 5: SAVE FINAL DOCUMENT ---
    output_filename = "RFP_Response.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_response_doc)
        
    print(f"\n✅ Success! The complete response has been saved to '{output_filename}'")

if __name__ == "__main__":
    main()