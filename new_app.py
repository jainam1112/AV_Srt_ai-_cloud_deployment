import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(page_title="Spiritual Archive Processor", layout="wide")

import json
from dotenv import load_dotenv
import os
from process_document import process_docx_with_openai
from timeline_component import render_timeline
from semantic_search_component import render_semantic_search

st.title("📄 Spiritual Narrative Document Processor")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in .env")
    st.stop()

uploaded_file = st.file_uploader("Upload .docx narrative document", type="docx", key="main_uploader")
use_sample = st.checkbox("Use sample output.json instead of processing")

if uploaded_file and not use_sample:
    if st.button("Process Document"):
        try:
            output = process_docx_with_openai(uploaded_file, api_key)
            with open("output.json", "w") as f:
                json.dump(output, f, indent=2)
            st.success("Processed successfully!")
        except Exception as e:
            st.error(str(e))

if use_sample or os.path.exists("output.json"):
    with open("output.json") as f:
        data = json.load(f)

    with st.expander("📁 Raw JSON"):
        st.json(data)

    render_timeline(data["TimelineData"])
    render_semantic_search(data["SemanticSearchIndexData"])
