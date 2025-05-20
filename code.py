import streamlit as st
import json
import os # For environment variables
from qdrant_client import QdrantClient, models # Import Qdrant client
from dotenv import load_dotenv # To load .env for Qdrant connection details

# --- Qdrant Configuration (Same as your main app, but for this annotator) ---
load_dotenv() # Load .env file if present for local development

QDRANT_HOST = os.getenv("QDRANT_HOST_ANNOTATOR", os.getenv("QDRANT_HOST", "localhost")) # Allow separate annotator host
QDRANT_PORT = int(os.getenv("QDRANT_PORT_ANNOTATOR", os.getenv("QDRANT_PORT", 6333)))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY_ANNOTATOR", os.getenv("QDRANT_API_KEY"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME_ANNOTATOR", os.getenv("COLLECTION_NAME", "gurudev_satsangs"))

qdrant_client = None
qdrant_error = None
try:
    qdrant_params = {"host": QDRANT_HOST, "port": QDRANT_PORT}
    if QDRANT_API_KEY:
        qdrant_params["api_key"] = QDRANT_API_KEY
        if not QDRANT_HOST.startswith("http") and QDRANT_HOST != "localhost":
            qdrant_params["https"] = True
    qdrant_client = QdrantClient(**qdrant_params)
    qdrant_client.get_collection(COLLECTION_NAME) # Test connection by trying to get the collection
except Exception as e:
    qdrant_error = f"Failed to connect to Qdrant: {e}. Please check connection details and ensure Qdrant is running."
    st.error(qdrant_error)


# --- Configuration: Define ALL your programmatic category keys (Same as before) ---
ALL_BIOGRAPHICAL_CATEGORY_KEYS = [
    "early_life_childhood", "education_learning", "spiritual_journey_influences",
    "professional_social_contributions", "travel_experiences",
    "meetings_notable_personalities", "hobbies_interests",
    "food_preferences_lifestyle", "family_personal_relationships",
    "health_wellbeing", "life_philosophy_core_values", "major_life_events",
    "legacy_impact", "miscellaneous_personal_details",
    "spiritual_training_discipleship", "ashram_infrastructure_development",
    "experiences_emotions", "organisation_events_milestones",
    "prophecy_future_revelations",
    "people_mentions_guidance", "people_mentions_praises",
    "people_mentions_general", "people_mentions_callouts",
    "pkd_relationship", "pkd_incidents", "pkd_stories", "pkd_references",
    "books_read", "books_recommended", "books_contributed_to",
    "books_references_general"
]

def format_key_for_display(key_name):
    return key_name.replace('_', ' ').title()

st.set_page_config(page_title="Qdrant Data Annotator", layout="wide")
st.title("📝 Gurudev Biographical Data Annotator (from Qdrant)")
st.caption("Load chunks from Qdrant, annotate, and generate fine-tuning data.")

# --- Initialize Session State ---
if 'current_annotations' not in st.session_state:
    st.session_state.current_annotations = {key: "" for key in ALL_BIOGRAPHICAL_CATEGORY_KEYS}
if 'transcript_chunk_input' not in st.session_state:
    st.session_state.transcript_chunk_input = ""
if 'qdrant_offset' not in st.session_state: # For pagination
    st.session_state.qdrant_offset = None
if 'loaded_qdrant_chunk_id' not in st.session_state:
    st.session_state.loaded_qdrant_chunk_id = None
if 'loaded_qdrant_chunk_text' not in st.session_state:
    st.session_state.loaded_qdrant_chunk_text = ""


# --- Qdrant Chunk Loader Section ---
st.sidebar.header("Load Chunk from Qdrant")

if qdrant_client and not qdrant_error:
    # Get list of unique transcript names for filtering (optional but helpful)
    @st.cache_data(ttl=300) # Cache for 5 minutes
    def get_transcript_names_from_qdrant():
        names = set()
        try:
            offset = None
            while True:
                points, next_offset = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME, limit=100, offset=offset,
                    with_payload=["transcript_name"], with_vectors=False
                )
                for point in points:
                    if point.payload and 'transcript_name' in point.payload:
                        names.add(point.payload['transcript_name'])
                if not next_offset or not points or len(names) > 500: break
                offset = next_offset
            return ["All Transcripts"] + sorted(list(names))
        except Exception as e:
            st.sidebar.warning(f"Could not fetch transcript names: {e}")
            return ["All Transcripts"]

    transcript_filter_name = st.sidebar.selectbox(
        "Filter by Transcript (Optional):",
        options=get_transcript_names_from_qdrant(),
        key="qdrant_transcript_filter"
    )

    search_term_qdrant = st.sidebar.text_input(
        "Search Qdrant for chunks (Optional):",
        placeholder="e.g., childhood, meditation",
        key="qdrant_search_term_input"
    )
    
    fetch_limit_qdrant = st.sidebar.slider("Number of chunks to browse:", 1, 20, 5, key="qdrant_fetch_limit")

    if st.sidebar.button("Load/Next Batch of Chunks", key="load_qdrant_chunks_button"):
        st.session_state.qdrant_offset = None # Reset offset for a new search/filter
        st.session_state.loaded_qdrant_chunk_id = None # Clear previously loaded chunk
        st.session_state.loaded_qdrant_chunk_text = ""
        st.session_state.transcript_chunk_input = "" # Clear manual input area
        # Clear previous annotations when loading new chunks
        st.session_state.current_annotations = {key: "" for key in ALL_BIOGRAPHICAL_CATEGORY_KEYS}


    # Fetch and display chunks for selection
    # This part runs whenever the page loads or relevant session state changes
    scroll_filter_qdrant = None
    if transcript_filter_name and transcript_filter_name != "All Transcripts":
        scroll_filter_qdrant = models.Filter(
            must=[models.FieldCondition(key="transcript_name", match=models.MatchValue(value=transcript_filter_name))]
        )
    
    points_to_display = []
    if qdrant_client: # Re-check if client is available
        try:
            if search_term_qdrant: # If there's a search term, use semantic search
                if 'openai_client_annotator' not in st.session_state: # Basic OpenAI client for embeddings
                    openai_api_key_annotator = os.getenv("OPENAI_API_KEY_ANNOTATOR", os.getenv("OPENAI_API_KEY"))
                    if openai_api_key_annotator:
                        from openai import OpenAI as OpenAIClientForAnnotator
                        st.session_state.openai_client_annotator = OpenAIClientForAnnotator(api_key=openai_api_key_annotator)
                    else:
                        st.sidebar.warning("OpenAI API Key needed for semantic search in annotator.")
                
                if 'openai_client_annotator' in st.session_state and st.session_state.openai_client_annotator:
                    query_vector = st.session_state.openai_client_annotator.embeddings.create(
                        input=[search_term_qdrant], model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
                    ).data[0].embedding
                    
                    hits = qdrant_client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_vector,
                        query_filter=scroll_filter_qdrant, # Apply transcript filter if selected
                        limit=fetch_limit_qdrant,
                        with_payload=True
                    )
                    points_to_display = hits # SearchResult objects
                else:
                    st.sidebar.warning("OpenAI client not initialized. Cannot perform semantic search.")

            else: # If no search term, scroll through chunks
                retrieved_points, next_page_offset = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=scroll_filter_qdrant,
                    limit=fetch_limit_qdrant,
                    offset=st.session_state.qdrant_offset,
                    with_payload=True, # We need 'original_text'
                    with_vectors=False # No need for vectors in annotator UI
                )
                points_to_display = retrieved_points # PointStruct objects
                st.session_state.qdrant_offset = next_page_offset # For "Load Next Batch"

            if points_to_display:
                st.sidebar.subheader("Select a Chunk to Annotate:")
                for point in points_to_display:
                    # SearchResult has .id and .payload, PointStruct also has .id and .payload
                    chunk_id = point.id
                    payload = point.payload
                    if payload and 'original_text' in payload:
                        text_snippet = payload['original_text'][:100] + "..." # Show a snippet
                        timestamp = payload.get('timestamp', 'N/A')
                        transcript_name_label = payload.get('transcript_name', 'Unknown')
                        
                        button_label = f"Load: {transcript_name_label} @ {timestamp} - \"{text_snippet}\""
                        if st.sidebar.button(button_label, key=f"load_chunk_{chunk_id}"):
                            st.session_state.transcript_chunk_input = payload['original_text']
                            st.session_state.loaded_qdrant_chunk_id = chunk_id
                            st.session_state.loaded_qdrant_chunk_text = payload['original_text']
                            # Clear previous annotations when loading a new specific chunk
                            st.session_state.current_annotations = {key: "" for key in ALL_BIOGRAPHICAL_CATEGORY_KEYS}
                            st.rerun() # Rerun to update the main text area and annotation fields
            elif st.session_state.get("load_qdrant_chunks_button_triggered"): # If button was pressed but no results
                 st.sidebar.info("No more chunks found with current filters/search or end of collection reached.")


        except Exception as e:
            st.sidebar.error(f"Error fetching chunks from Qdrant: {e}")
            print(f"Error fetching from qdrant: {e}") # For terminal debugging

elif qdrant_error:
    st.sidebar.error(qdrant_error)
else:
    st.sidebar.warning("Qdrant client not initialized. Cannot load chunks.")


# --- Main Annotation Area (largely same as before) ---
st.header("1. Transcript Chunk for Annotation")
if st.session_state.loaded_qdrant_chunk_id:
    st.info(f"Loaded chunk ID from Qdrant: {st.session_state.loaded_qdrant_chunk_id}")
    # Display the loaded text but allow edits if needed, or make it read-only
    transcript_chunk = st.text_area(
        "Current transcript segment (loaded from Qdrant):",
        value=st.session_state.transcript_chunk_input, # This is updated when a Qdrant chunk is loaded
        height=250,
        key="transcript_chunk_input_area_main",
        # disabled=True # Optionally make it non-editable if only annotating loaded chunks
    )
    # Ensure session state is updated if user somehow edits it (if not disabled)
    st.session_state.transcript_chunk_input = transcript_chunk

else:
    transcript_chunk = st.text_area(
        "Paste or type the transcript segment here (or load from Qdrant using sidebar):",
        value=st.session_state.transcript_chunk_input,
        height=250,
        key="transcript_chunk_input_area_manual"
    )
    st.session_state.transcript_chunk_input = transcript_chunk


st.header("2. Extract Verbatim Quotes for Each Category")
st.markdown("""
**Instructions:**
- For each category, copy **EXACT verbatim quote(s)** from the chunk above.
- If **multiple quotes** fit one category, separate them by a **NEWLINE** within that category's box.
- If no text fits, leave the box **empty**.
""")

annotation_inputs = {}
cols_per_row = 2
category_columns = st.columns(cols_per_row)
col_idx = 0
for i, key in enumerate(ALL_BIOGRAPHICAL_CATEGORY_KEYS):
    with category_columns[col_idx]:
        display_label = f"{i+1}. {format_key_for_display(key)}"
        annotation_inputs[key] = st.text_area(
            display_label,
            height=100,
            key=f"category_{key}",
            placeholder=f"Quotes for {format_key_for_display(key)}...",
            value=st.session_state.current_annotations.get(key, "") # Use .get for safety
        )
        st.session_state.current_annotations[key] = annotation_inputs[key]
    col_idx = (col_idx + 1) % cols_per_row

st.header("3. Generate Fine-Tuning Example (JSONL Format)")
if st.button("Generate JSONL Line", key="generate_button_main"):
    current_chunk_text_for_jsonl = st.session_state.transcript_chunk_input # Use current content of text area
    if not current_chunk_text_for_jsonl.strip():
        st.warning("Please load or provide a transcript chunk.")
    else:
        assistant_json_data = {}
        for key, text_area_content in st.session_state.current_annotations.items(): # Use session state directly
            if text_area_content and text_area_content.strip():
                quotes_list = [quote.strip() for quote in text_area_content.split('\n') if quote.strip()]
                assistant_json_data[key] = quotes_list
            else:
                assistant_json_data[key] = []

        assistant_content_str = f"```json\n{json.dumps(assistant_json_data, indent=2)}\n```"
        user_message_content = f"Transcript Segment: \"{current_chunk_text_for_jsonl.strip()}\""
        jsonl_line_data = {
            "messages": [
                {"role": "user", "content": user_message_content},
                {"role": "assistant", "content": assistant_content_str}
            ]
        }
        st.subheader("Generated JSONL Line (Copy this):")
        st.code(json.dumps(jsonl_line_data), language="json")
        st.info("Copy and paste this line into your .jsonl training file.")

        # Option to clear ONLY annotations for the current chunk, keeping the chunk text
        if st.button("Clear Annotations (Keep Chunk Text)", key="clear_annotations_only_button"):
            st.session_state.current_annotations = {key: "" for key in ALL_BIOGRAPHICAL_CATEGORY_KEYS}
            st.rerun()

st.markdown("---")
st.markdown("Collect many diverse examples for effective fine-tuning!") # type: ignore