import streamlit as st
import json
import os # For environment variables
from qdrant_client import QdrantClient, models # Import Qdrant client
from dotenv import load_dotenv # To load .env for Qdrant connection details
import re # Added for parsing reranked indices
from openai import OpenAIError # Added for OpenAI error handling

# --- Qdrant Configuration (Same as your main app, but for this annotator) ---
load_dotenv() # Load .env file if present for local development

QDRANT_HOST = os.getenv("QDRANT_HOST_ANNOTATOR", os.getenv("QDRANT_HOST", "localhost")) # Allow separate annotator host
QDRANT_PORT = int(os.getenv("QDRANT_PORT_ANNOTATOR", os.getenv("QDRANT_PORT", 6333)))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY_ANNOTATOR", os.getenv("QDRANT_API_KEY"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME_ANNOTATOR", os.getenv("COLLECTION_NAME", "gurudev_satsangs"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") # Ensure this is defined
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4-turbo-preview") # Added
ANSWER_EXTRACTION_MODEL = os.getenv("ANSWER_EXTRACTION_MODEL", "gpt-3.5-turbo") # Added

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
                if not next_offset or not points or len(names) > 500: break # Stop if no more points or enough names
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
        # Note: No st.rerun() here, data fetching happens below in the same script run


# Fetch and display chunks for selection
# This part runs whenever the page loads or relevant session state changes

    # Build Qdrant filter conditions
    all_must_conditions = []
    all_must_not_conditions = []

    # Transcript name filter
    if transcript_filter_name and transcript_filter_name != "All Transcripts":
        all_must_conditions.append(
            models.FieldCondition(key="transcript_name", match=models.MatchValue(value=transcript_filter_name))
        )

    scroll_filter_qdrant = None
    filter_args = {}
    if all_must_conditions:
        filter_args["must"] = all_must_conditions
    if all_must_not_conditions:
        filter_args["must_not"] = all_must_not_conditions
    
    if filter_args:
        scroll_filter_qdrant = models.Filter(**filter_args)

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
                for point_idx, point in enumerate(points_to_display): # Added enumerate for unique keys
                    # SearchResult has .id and .payload, PointStruct also has .id and .payload
                    chunk_id = point.id
                    payload = point.payload
                    if payload and 'original_text' in payload: # This check is good
                        text_snippet = payload['original_text'][:100] + "..." # Show a snippet
                        timestamp = payload.get('timestamp', 'N/A')
                        transcript_name_label = payload.get('transcript_name', 'Unknown')
                        
                        expander_label = f"{transcript_name_label} @ {timestamp} - \"{text_snippet}\""
                        
                        with st.sidebar.expander(expander_label, expanded=(point_idx == 0)):
                            st.caption("Full Text Preview:")
                            # Displaying the full payload['original_text']
                            st.markdown(f"```\n{payload['original_text']}\n```")

                            button_key = f"qcl_add_to_annotate_{chunk_id}_{point_idx}" # Unique key for this button
                            if st.button("Add to Annotate", key=button_key): # Button inside the expander
                                # --- Button Click Debugging ---
                                print(f"--- QCL DEBUG: 'Add to Annotate' button clicked for chunk_id: {chunk_id} ---") # Terminal debug
                                st.session_state.qcl_debug_button_click = f"QCL 'Add to Annotate' CLICKED for chunk {chunk_id}. Original text snippet: '{payload.get('original_text', 'N/A')[:30]}...'"
                                # --- End Button Click Debugging ---

                                loaded_text = payload.get('original_text') # Get the text
                                if loaded_text is None: # Ensure it's a string, even if original_text was None
                                    loaded_text = ""
                                    # Simple debug message (can be part of the main debug message if needed)
                                    # st.session_state.debug_message = f"Button for chunk {chunk_id} clicked. Original text was None."
                                # else:
                                    # Simple debug message
                                    # st.session_state.debug_message = f"Button for chunk {chunk_id} clicked. Text length: {len(loaded_text)}. Starting with: '{loaded_text[:30]}...'"

                                st.session_state.transcript_chunk_input = loaded_text
                                st.session_state.loaded_qdrant_chunk_id = chunk_id
                                st.session_state.loaded_qdrant_chunk_text = loaded_text # Keep this consistent
                                
                                # Clear previous annotations when loading a new specific chunk
                                st.session_state.current_annotations = {key: "" for key in ALL_BIOGRAPHICAL_CATEGORY_KEYS}
                                # st.session_state.debug_message += f" | Set loaded_qdrant_chunk_id to {chunk_id}. transcript_chunk_input (start): '{loaded_text[:20]}...'. Rerunning." # Simplified above
                                st.rerun() # Rerun to update the main text area and annotation fields
            
            elif not points_to_display and st.session_state.get("load_qdrant_chunks_button_triggered"): # Check if button was pressed AND no points
                 st.sidebar.info("No chunks found with current filters/search or end of collection reached.")
                 st.session_state.load_qdrant_chunks_button_triggered = False # Reset trigger


        except Exception as e:
            st.sidebar.error(f"Error fetching chunks from Qdrant: {e}")
            print(f"Error fetching from qdrant: {e}") # For terminal debugging

elif qdrant_error:
    st.sidebar.error(qdrant_error)
else:
    st.sidebar.warning("Qdrant client not initialized. Cannot load chunks.")

# --- Semantic Search & Analysis UI (Moved to Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Semantic Search & Analysis")

# Ensure OpenAI client is available for search, rerank, and answer extraction
if 'openai_client_annotator' not in st.session_state:
    openai_api_key_annotator = os.getenv("OPENAI_API_KEY_ANNOTATOR", os.getenv("OPENAI_API_KEY"))
    if openai_api_key_annotator:
        from openai import OpenAI as OpenAIClientForAnnotator # Ensure client is imported
        st.session_state.openai_client_annotator = OpenAIClientForAnnotator(api_key=openai_api_key_annotator)
    else:
        st.sidebar.warning("OpenAI API Key not found. Semantic search and LLM features will be unavailable.")

# Placeholder for extract_answer_span function (as definition was not provided)
def extract_answer_span(query: str, context: str, model: str) -> str:
    """
    Placeholder function to extract an answer span using an LLM.
    Replace with your actual implementation.
    """
    # This is a dummy implementation.
    # In a real scenario, you would make an API call to an LLM.
    # For example, using st.session_state.openai_client_annotator.chat.completions.create(...)
    print(f"Placeholder: Called extract_answer_span for query '{query}' with model '{model}'")
    if query.lower() in context.lower():
        # A very naive "extraction"
        start_index = context.lower().find(query.lower())
        end_index = start_index + len(query)
        return context[start_index:end_index]
    return f"Could not extract specific span for '{query}' from context (placeholder response)."


if 'openai_client_annotator' in st.session_state and st.session_state.openai_client_annotator and qdrant_client:
    search_query_input = st.sidebar.text_input("Enter your search query:", key="main_search_query_input")

    # Search Options Columns
    # For sidebar, columns might not be ideal, stacking them:
    use_llm_reranking = st.sidebar.checkbox("Enable LLM Reranking", value=True, key="rerank_toggle_checkbox_sidebar",
                                     help=f"Uses {RERANK_MODEL} to reorder results.")
    do_pinpoint_answer_extraction = st.sidebar.checkbox("Pinpoint Answer Snippet", value=True, key="pinpoint_answer_checkbox_sidebar",
                                              help=f"Uses {ANSWER_EXTRACTION_MODEL} to identify relevant parts.")
    initial_search_results_limit = st.sidebar.slider("Initial results to retrieve:", min_value=3, max_value=30, value=5, key="search_results_limit_sidebar_slider",
                                              help="Number of chunks from Qdrant before reranking/pinpointing.")

    custom_reranking_instructions_input = ""
    if use_llm_reranking:
        custom_reranking_instructions_input = st.sidebar.text_area(
            "Custom Reranking Instructions (Optional):",
            placeholder="e.g., 'Prioritize practical advice.'",
            key="custom_rerank_instructions_input_area_sidebar",
            height=100
        )

    # --- Search and Display Results (in Sidebar) ---
    if search_query_input:
        with st.spinner("Searching and analyzing results..."): # Changed from st.sidebar.spinner
            # 1. Get Query Embedding
            try:
                query_embedding_response = st.session_state.openai_client_annotator.embeddings.create(model=EMBEDDING_MODEL, input=[search_query_input])
                query_vector_for_search = query_embedding_response.data[0].embedding
            except OpenAIError as e:
                st.sidebar.error(f"Failed to get embedding: {e}")
                # st.stop() # Consider removing st.stop() if it's too disruptive in sidebar
                query_vector_for_search = None
            except Exception as e:
                st.sidebar.error(f"Unexpected error preparing query: {e}")
                # st.stop()
                query_vector_for_search = None

            if query_vector_for_search:
                # 2. Search Qdrant
                try:
                    print(f"Sidebar: Searching Qdrant for query: '{search_query_input}' with limit {initial_search_results_limit}")
                    qdrant_search_hits = qdrant_client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_vector_for_search,
                        limit=initial_search_results_limit,
                        with_payload=True
                    )
                except Exception as e:
                    st.sidebar.error(f"Error searching Qdrant: {e}")
                    # st.stop()
                    qdrant_search_hits = []
                
                retrieved_results_payloads = [hit.payload for hit in qdrant_search_hits if hit.payload]

                if not retrieved_results_payloads:
                    st.sidebar.info("No results found for your query.")
                    # st.stop()

                # 3. Optional LLM Reranking
                if use_llm_reranking and len(retrieved_results_payloads) > 1:
                    with st.spinner(f"Reranking with {RERANK_MODEL}..."): # Changed from st.sidebar.spinner
                        print(f"Sidebar: Attempting to rerank {len(retrieved_results_payloads)} results.")
                        excerpts_for_llm_rerank = []
                        for i, res_payload in enumerate(retrieved_results_payloads):
                            excerpts_for_llm_rerank.append(f"[{i+1}] {res_payload.get('original_text', 'Error: Text not found')}")
                        
                        rerank_instruction_prefix = "Rerank text excerpts by relevance to the user's query."
                        if custom_reranking_instructions_input:
                            rerank_instruction_prefix += f" Criteria: {custom_reranking_instructions_input}."
                        
                        llm_rerank_prompt = (
                            f"{rerank_instruction_prefix}\n\n"
                            f"User's query: \"{search_query_input}\"\n\n"
                            f"Excerpts (prefixed with index in brackets, e.g., [1]):\n"
                            f"{chr(10).join(excerpts_for_llm_rerank)}\n\n"
                            f"Return *only* a comma-separated list of original indices (e.g., '3,1,2'), "
                            f"most relevant to least. No other text."
                        )
                        
                        try:
                            rerank_response = st.session_state.openai_client_annotator.chat.completions.create(
                                model=RERANK_MODEL,
                                messages=[
                                    {"role": "system", "content": "You rerank search results based on queries and instructions."},
                                    {"role": "user", "content": llm_rerank_prompt}
                                ],
                                max_tokens=200,
                                temperature=0.0
                            )
                            reranked_indices_str_output = rerank_response.choices[0].message.content.strip()
                            print(f"Sidebar: Reranker LLM output: '{reranked_indices_str_output}'")

                            parsed_reranked_indices = []
                            seen_indices_for_rerank = set()
                            if reranked_indices_str_output:
                                raw_indices_from_llm = re.findall(r'\d+', reranked_indices_str_output)
                                for s_idx in raw_indices_from_llm:
                                    try:
                                        val = int(s_idx) - 1  # 0-based index
                                        if 0 <= val < len(retrieved_results_payloads) and val not in seen_indices_for_rerank:
                                            parsed_reranked_indices.append(val)
                                            seen_indices_for_rerank.add(val)
                                    except ValueError:
                                        print(f"Sidebar: Reranker LLM non-integer index: '{s_idx}'")
                            
                            if len(parsed_reranked_indices) > 0:
                                original_indices_set = set(range(len(retrieved_results_payloads)))
                                missing_indices_after_rerank = sorted(list(original_indices_set - seen_indices_for_rerank))
                                final_indices_order_after_rerank = parsed_reranked_indices + missing_indices_after_rerank
                                
                                retrieved_results_payloads = [retrieved_results_payloads[i] for i in final_indices_order_after_rerank if 0 <= i < len(retrieved_results_payloads)]
                                st.sidebar.caption(f"ℹ️ Results reranked by {RERANK_MODEL}.")
                            else:
                                st.sidebar.warning("Reranking failed or no reordering. Using original order.", icon="⚠️")
                                print("Sidebar: Reranking failed or no reordering.")

                        except OpenAIError as e_rerank:
                            st.sidebar.warning(f"LLM Reranking API error: {e_rerank}. Using original order.", icon="⚠️")
                            print(f"Sidebar: OpenAI API error during reranking: {e_rerank}")
                        except Exception as e_rerank_general:
                            st.sidebar.warning(f"Error during reranking: {e_rerank_general}. Using original order.", icon="⚠️")
                            print(f"Sidebar: Unexpected error during reranking: {e_rerank_general}")
                
                # 4. Display Results in Sidebar
                st.sidebar.subheader(f"Search Results for: \"{search_query_input}\"")
                if not retrieved_results_payloads:
                    st.sidebar.info("No relevant information found.")
                else:
                    for i, result_payload_data in enumerate(retrieved_results_payloads, 1):
                        if not isinstance(result_payload_data, dict):
                            st.sidebar.error(f"Result {i} invalid payload.")
                            print(f"Sidebar: Invalid payload for result {i}: {result_payload_data}")
                            continue

                        full_chunk_text_content = result_payload_data.get('original_text', 'Error: Text not found')
                        
                        expander_title = f"Result {i} @ {result_payload_data.get('timestamp', 'N/A')} | {result_payload_data.get('transcript_name', 'N/A')}"
                        with st.sidebar.expander(expander_title, expanded=(i==1)):
                            if do_pinpoint_answer_extraction and full_chunk_text_content != 'Error: Text not found':
                                with st.spinner(f"Extracting answer for result {i}..."): # This was already correct
                                    pinpointed_answer_span = extract_answer_span(search_query_input, full_chunk_text_content, ANSWER_EXTRACTION_MODEL)
                                
                                if pinpointed_answer_span and \
                                   pinpointed_answer_span.strip() and \
                                   pinpointed_answer_span.strip().lower() != full_chunk_text_content.strip().lower() and \
                                   len(pinpointed_answer_span.strip()) < len(full_chunk_text_content.strip()):
                                    st.markdown(f"🎯 **Pinpointed Snippet:**")
                                    st.markdown(f"> *{pinpointed_answer_span.strip()}*")
                                    st.markdown("--- \n**Full Context:**")
                                elif pinpointed_answer_span and pinpointed_answer_span.strip().lower() == full_chunk_text_content.strip().lower():
                                     st.caption(f"(Full chunk identified as most direct by {ANSWER_EXTRACTION_MODEL})")

                            st.markdown(f"{full_chunk_text_content}")
                            
                            entities_data = result_payload_data.get('entities')
                            if entities_data and isinstance(entities_data, dict):
                                entity_parts = []
                                if entities_data.get('people'): entity_parts.append(f"**People:** {', '.join(entities_data['people'])}")
                                if entities_data.get('places'): entity_parts.append(f"**Places:** {', '.join(entities_data['places'])}")
                                if 'self_references' in entities_data: entity_parts.append(f"**Self-ref:** {entities_data.get('self_references', 'N/A')}")
                                if entity_parts:
                                    st.markdown("---")
                                    st.markdown(" ".join(entity_parts), unsafe_allow_html=True)
                            
                            # Add a button to load this chunk into the main annotation area
                            if st.button(f"Load Chunk {i} for Annotation", key=f"sidebar_load_chunk_{result_payload_data.get('id', i)}"):
                                st.session_state.transcript_chunk_input = full_chunk_text_content
                                st.session_state.loaded_qdrant_chunk_id = result_payload_data.get('id', f"searched_result_{i}")
                                st.session_state.loaded_qdrant_chunk_text = full_chunk_text_content
                                st.session_state.current_annotations = {key: "" for key in ALL_BIOGRAPHICAL_CATEGORY_KEYS}
                                st.rerun()
elif not qdrant_client:
    st.sidebar.warning("Qdrant client not available. Search disabled.")
else: # OpenAI client not available
    st.sidebar.warning("OpenAI client not available. Semantic search and LLM features disabled.")


# --- Main Annotation Area (largely same as before) ---
st.header("1. Transcript Chunk for Annotation")

# Display debug message from QCL button if any
if "qcl_debug_button_click" in st.session_state:
    st.error(f"DEBUG FROM QCL BUTTON CLICK: {st.session_state.qcl_debug_button_click}") # Use st.error to make it stand out
    del st.session_state.qcl_debug_button_click # Clear after displaying

# Display general debug message if any (you might have this from other parts)
if "debug_message" in st.session_state: 
    st.info(f"Debug: {st.session_state.debug_message}")
    del st.session_state.debug_message 

# More debug output for main area
st.warning(f"Main Area Checkpoint (after potential rerun): `loaded_qdrant_chunk_id` = `{st.session_state.get('loaded_qdrant_chunk_id')}`")
st.warning(f"Main Area Checkpoint (after potential rerun): `transcript_chunk_input` (first 70 chars) = `{str(st.session_state.get('transcript_chunk_input', ''))[:70]}`")
st.warning(f"Main Area Checkpoint (after potential rerun): `loaded_qdrant_chunk_text` (first 70 chars) = `{str(st.session_state.get('loaded_qdrant_chunk_text', ''))[:70]}`")


if st.session_state.get('loaded_qdrant_chunk_id'): # Use .get for safety, though it should be set
    st.success(f"Main Area: Rendering text area for loaded chunk ID: {st.session_state.loaded_qdrant_chunk_id}")
    st.write(f"Main Area (inside if): `st.session_state.transcript_chunk_input` to be used for text_area value (first 70 chars): `{str(st.session_state.get('transcript_chunk_input', ''))[:70]}`")
    
    # Make the key dynamic based on the loaded chunk ID
    text_area_key = f"transcript_chunk_input_area_main_{st.session_state.loaded_qdrant_chunk_id}"
    
    transcript_chunk = st.text_area(
        "Current transcript segment (loaded from Qdrant):",
        value=st.session_state.transcript_chunk_input, # This is updated when a Qdrant chunk is loaded
        height=250,
        key=text_area_key, # Use the dynamic key
        # disabled=True # Optionally make it non-editable if only annotating loaded chunks
    )
    # Ensure session state is updated if user somehow edits it (if not disabled)
    if transcript_chunk != st.session_state.transcript_chunk_input: # Update if changed
        st.session_state.transcript_chunk_input = transcript_chunk


else:
    transcript_chunk = st.text_area(
        "Paste or type the transcript segment here (or load from Qdrant using sidebar):",
        value=st.session_state.transcript_chunk_input,
        height=250,
        key="transcript_chunk_input_area_manual"
    )
    if transcript_chunk != st.session_state.transcript_chunk_input: # Update if changed
        st.session_state.transcript_chunk_input = transcript_chunk


st.header("2. Extract Verbatim Quotes for Each Category")
st.markdown("""
**Instructions:**
- For each category, copy **EXACT verbatim quote(s)** from the chunk above.
- If **multiple quotes** fit one category, separate them by a **NEWLINE** within that category's box.
- If no text fits, leave the box **empty**.
""")

annotation_inputs = {} # Not strictly needed if directly updating session state
cols_per_row = 2
category_columns = st.columns(cols_per_row)
col_idx = 0
for i, key in enumerate(ALL_BIOGRAPHICAL_CATEGORY_KEYS):
    with category_columns[col_idx]:
        display_label = f"{i+1}. {format_key_for_display(key)}"
        # Directly update session_state in the text_area's on_change or rely on Streamlit's execution model
        current_val = st.session_state.current_annotations.get(key, "")
        new_val = st.text_area(
            display_label,
            height=100,
            key=f"category_{key}",
            placeholder=f"Quotes for {format_key_for_display(key)}...",
            value=current_val
        )
        if new_val != current_val: # Update session state if value changed
            st.session_state.current_annotations[key] = new_val
            # No st.rerun() needed here, will update on next interaction or button press
    col_idx = (col_idx + 1) % cols_per_row

st.header("3. Generate Fine-Tuning Example (JSONL Format)")
if st.button("Generate JSONL Line", key="generate_button_main"):
    current_chunk_text_for_jsonl = st.session_state.transcript_chunk_input # Use current content of text area
    if not current_chunk_text_for_jsonl.strip():
        st.error("Cannot generate JSONL line. The current chunk text is empty.")
    else:
        assistant_json_data = {}
        for key, text_area_content in st.session_state.current_annotations.items(): # Use session state directly
            if text_area_content and text_area_content.strip():
                quotes_list = [quote.strip() for quote in text_area_content.split('\n') if quote.strip()]
                assistant_json_data[key] = quotes_list
            else:
                assistant_json_data[key] = [] # Ensure all keys are present, even if empty

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