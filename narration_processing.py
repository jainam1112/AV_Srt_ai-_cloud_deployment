import streamlit as st
import json
import os
import openai
import docx
from docx import Document
from dotenv import load_dotenv, find_dotenv

# --- 1. INITIAL APP CONFIGURATION & SETUP ---

st.set_page_config(
    page_title="Document Processing Prototype",
    layout="centered"
)

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI client
try:
    api_key = OPENAI_API_KEY
    client = openai.OpenAI(api_key=api_key)
except KeyError:
    st.error("🔴 OPENAI_API_KEY not found! Please create a .env file and add your OpenAI API key to it.")
    st.stop()

MODEL_NAME = "gpt-4-1106-preview"

# --- 2. THE COMPREHENSIVE LLM SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are an expert data analyst and archivist specializing in natural language processing, entity recognition, and semantic structuring. You have been tasked with processing a narrative document about the life of a spiritual master to create a structured, searchable digital archive. Your goal is to transform unstructured text into a meticulously organized JSON output. Adhere to the following instructions with absolute precision.

### 1. Core Mission
Process the document to: 1) Translate Gujarati, preserving key terms. 2) Extract and standardize entities (People, Places, Dates with rationale for approximations). 3) Categorize and summarize incidents. 4) Extract a representative, fully-cited text snippet for each incident. 5) Compile all information into a single, structured JSON output.

### 2. Key Terms to Preserve
`prasad`, `atmarpit`, `Vachanamrutji`, `Tattvajnaan`, `Patrank`, `satsang`, `Mumukshu`, `Param Krupalu Dev`, `Prabhu Shriji`, `Shrimad Rajchandra`, `seva`, `Pramad`, `Bhakti`, `Pratishtha`, `Vanaprastha Diksha`, `Updeshamrut`.

### 3. Thematic Categories
`Spiritual Guidance`, `Divine Experiences`, `Service (Seva)`, `Personal Milestones`, `Teachings Analysis`, `Travels`, `Community Impact`, `Guru Param Krupalu Dev Relation`.

### 4. Snippet Formatting
"[Exact quoted text, 300-500 characters.]"  
— {Year}, {Standardized Place} (Document: [Original Document Name], Page: [Page Number])  
- Narrator: [Narrator's Name, if identified]

### 5. Final Output Specification
Compile all processed information into a single, comprehensive JSON object. The structure must be exactly as follows, with 'TimelineData' sorted chronologically (Undated events at the end, alphabetized by Summary).

```json
{
  "TimelineData": [
    {
      "UID": "event_001", "StandardizedDate": "1993", "DateFootnote": "Optional note", "Summary": "Summary.",
      "ThematicCategories": ["Category1"], "AssociatedPeople": ["Person1"], "AssociatedPlaces": ["Place1"],
      "FormattedSnippet": "Formatted snippet text...", "SourceDocument": "doc_name.docx", "SourcePage": 4
    }
  ],
  "SemanticSearchIndexData": [
    {
      "Content": "Snippet text for embedding...",
      "Metadata": { "UID": "event_001", "Date": "1993", "Themes": ["..."], "People": ["..."], "Places": ["..."], "Source": "..." }
    }
  ],
  "GlossaryData": [
    { "Term": "prasad", "Definition": "A devotional offering..." }
  ]
}
```

Begin processing the provided document now.
"""

# --- 3. CORE PROCESSING FUNCTION WITH CACHING ---
@st.cache_data(show_spinner="Reading and processing document with AI...")
def process_document_with_llm(uploaded_file):
    doc_name = uploaded_file.name

    # Read DOCX content
    try:
        doc = Document(uploaded_file)
        document_text = '\n'.join([para.text for para in doc.paragraphs])
        if not document_text.strip():
            st.error("The uploaded document appears to be empty.")
            return None
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return None

    # Prepare the message to LLM
    user_content = f"Here is the document to process. Document Name: {doc_name}\n\n--- DOCUMENT CONTENT ---\n\n{document_text}"

    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        response_content = response.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        st.error(f"An error occurred while communicating with the OpenAI API: {e}")
        return None

# --- 4. STREAMLIT UI ---
st.title("📄 Document to Structured JSON Processor")
st.markdown("This app uses a sophisticated LLM prompt to convert a narrative document into a structured JSON file for the Spiritual Master Archive project.")

uploaded_file1 = st.file_uploader("Upload a file", key="file1")
uploaded_file2 = st.file_uploader("Upload a file", key="file2")

if uploaded_file1 is not None:
    if st.button("Process Document", type="primary"):
        structured_data = process_document_with_llm(uploaded_file1)
        
        if structured_data:
            st.success("Document processed successfully!")
            
            # --- START: NEW CODE FOR SAVING ---
            
            # 1. Convert the dictionary to a JSON formatted string for downloading
            json_string = json.dumps(structured_data, indent=2, ensure_ascii=False)
            
            # 2. Create a dynamic filename based on the uploaded file
            output_filename = f"{os.path.splitext(uploaded_file1.name)[0]}_output.json"
            
            # 3. Add a download button
            st.download_button(
                label="📥 Download JSON Output",
                data=json_string,
                file_name=output_filename,
                mime="application/json",
            )
            
            # --- END: NEW CODE FOR SAVING ---

            st.header("Structured JSON Output")
            st.json(structured_data) # Display the JSON in the app as before
        else:
            st.error("Failed to process the document. Please check the error messages above.")
else:
    st.info("Please upload a document to see the processing in action.")
