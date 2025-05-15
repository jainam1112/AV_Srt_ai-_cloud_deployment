#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gurudev Satsang Transcript Processing Pipeline + Streamlit UI
- SRT Upload and Processing
- Chunking with Overlap
- Optional Claude Entity Tagging
- OpenAI Embeddings
- Qdrant Vector Storage
- Semantic Search
- LLM-based Reranking with Custom User Instructions
- LLM-based Answer Span Extraction
- Displaying a List of Already Processed Transcripts
"""

import os
import uuid
import srt
import httpx
import json
import re
import streamlit as st
from qdrant_client import QdrantClient, models
from pathlib import Path
from openai import OpenAI, OpenAIError # Ensure OpenAIError is imported
from dotenv import load_dotenv, find_dotenv
import tempfile
import logging

# === CONFIG ===
load_dotenv(find_dotenv()) # Load .env file if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # For Qdrant Cloud
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gurudev_satsangs")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536 # Corresponds to text-embedding-3-small
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 20))

ENTITY_MODEL = os.getenv("ENTITY_MODEL", "claude-3-opus-20240229")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4-turbo-preview")
ANSWER_EXTRACTION_MODEL = os.getenv("ANSWER_EXTRACTION_MODEL", "gpt-3.5-turbo")
# FINE_TUNED_BIO_MODEL_ID = os.getenv("FINE_TUNED_BIO_MODEL_ID") # For later integration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === INITIALIZATION ===
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it as an environment variable or in Streamlit secrets.")
    st.stop()

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    logger.error(f"OpenAI client initialization error: {e}")
    st.stop()

try:
    if QDRANT_HOST.startswith("http"):
        qdrant_params = {"url": QDRANT_HOST}
    else:
        qdrant_params = {"host": QDRANT_HOST, "port": QDRANT_PORT}
    
    if QDRANT_API_KEY:
        qdrant_params["api_key"] = QDRANT_API_KEY
        if not QDRANT_HOST.startswith("http") and QDRANT_HOST != "localhost":
            qdrant_params["https"] = True  # Changed from https_ to https

    qdrant = QdrantClient(**qdrant_params)
    qdrant.get_collections() # Test connection
    logger.info(f"Connected to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    st.error(f"Qdrant connection error at {QDRANT_HOST}:{QDRANT_PORT}. Details: {e}")
    logger.error(f"Qdrant connection error: {e}")
    st.stop()
# Ensure collection exists
try:
    collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
    if collection_info.config.params.vectors.size != EMBEDDING_DIM or \
       collection_info.config.params.vectors.distance != models.Distance.COSINE:
        logger.warning(f"Collection '{COLLECTION_NAME}' exists with different configuration. Recreating.")
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
        )
        logger.info(f"Recreated Qdrant collection: {COLLECTION_NAME}")
except Exception: # Covers collection not found and other Qdrant client errors
    logger.info(f"Collection '{COLLECTION_NAME}' not found or error accessing. Creating/Recreating.")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
    )
    logger.info(f"Ensured Qdrant collection '{COLLECTION_NAME}' exists with correct config.")


# === HELPERS ===
def parse_srt_file(file_path: str) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return list(srt.parse(f.read()))
    except Exception as e:
        logger.error(f"Error parsing SRT file {file_path}: {e}")
        st.error(f"Failed to parse SRT file: {e}")
        return []

def srt_to_chunks(subs: list, chunk_size_words: int, overlap_words: int) -> list:
    chunks, current_sub_idx = [], 0
    if not subs: return []
    while current_sub_idx < len(subs):
        buffer, word_count, temp_idx = [], 0, current_sub_idx
        while temp_idx < len(subs):
            sub = subs[temp_idx]
            words_in_sub = len(sub.content.split())
            # Allow slight overrun for the last sub in a chunk to avoid breaking sentences awkwardly
            if not buffer or word_count + words_in_sub <= chunk_size_words + (chunk_size_words * 0.2):
                buffer.append(sub)
                word_count += words_in_sub
            else:
                break
            temp_idx += 1
        if not buffer: break
        chunks.append({'text': ' '.join(s.content for s in buffer), 'timestamp': str(buffer[0].start)})
        
        if overlap_words <= 0 or len(buffer) <= 1:
            current_sub_idx = temp_idx
        else:
            retained_words_for_overlap, subs_in_overlap_count = 0, 0
            # Iterate backwards from the end of the current chunk's subtitles
            for i in range(len(buffer) - 1, 0, -1): # Start from second to last sub in buffer
                retained_words_for_overlap += len(buffer[i].content.split())
                subs_in_overlap_count += 1
                if retained_words_for_overlap >= overlap_words:
                    break
            
            subs_to_advance_from_chunk_start = max(1, len(buffer) - subs_in_overlap_count)
            next_start_idx = current_sub_idx + subs_to_advance_from_chunk_start

            if next_start_idx <= current_sub_idx : # Safety: ensure progress
                current_sub_idx = temp_idx # Fallback to no overlap for this step
            else:
                current_sub_idx = min(next_start_idx, len(subs)) # Don't go out of bounds
    logger.info(f"Created {len(chunks)} chunks from SRT. Target size: {chunk_size_words} words, overlap: {overlap_words} words.")
    return chunks

def tag_entities_with_claude(text: str) -> dict:
    if not CLAUDE_API_KEY:
        # st.sidebar.warning("CLAUDE_API_KEY not set. Skipping entity tagging.", icon="⚠️") # Moved to processing section
        return {}
    headers = {"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    prompt = (f"Analyze text, extract entities. Respond *only* with JSON: {{'people': [], 'places': [], 'self_references': bool}}.\nText:\n{text}\nJSON Output:")
    body = {"model": ENTITY_MODEL, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}
    try:
        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(CLAUDE_API_URL, headers=headers, json=body)
            response.raise_for_status()
        data = response.json()
        if data.get('content') and data['content'][0].get('type') == 'text':
            raw_text = data['content'][0]['text']
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
            json_str = json_match.group(1) if json_match else raw_text[raw_text.find('{'):raw_text.rfind('}')+1]
            try:
                entities = json.loads(json_str)
                return {
                    'people': entities.get('people', []) if isinstance(entities.get('people'), list) else [],
                    'places': entities.get('places', []) if isinstance(entities.get('places'), list) else [],
                    'self_references': entities.get('self_references', False) if isinstance(entities.get('self_references'), bool) else False
                }
            except json.JSONDecodeError as e:
                logger.error(f"Claude JSON parse error: {e}. Response: {json_str[:100]}"); return {}
        return {}
    except httpx.HTTPStatusError as e:
        logger.error(f"Claude API HTTP error: {e.response.status_code} - {e.response.text}"); return {}
    except Exception as e:
        logger.error(f"Claude API/tagging error: {e}"); return {}

def batch_get_embeddings(texts: list) -> list:
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([obj.embedding for obj in response.data])
        except OpenAIError as e:
            logger.error(f"OpenAI embedding error: {e}");
            st.error(f"OpenAI API error during embedding: {e}. Some chunks may not be embedded.")
            all_embeddings.extend([None] * len(batch))
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            st.error(f"Unexpected error during embedding: {e}. Some chunks may not be embedded.")
            all_embeddings.extend([None] * len(batch))
    return all_embeddings

def upsert_to_qdrant(points_to_upsert: list):
    if not points_to_upsert: return
    # Filter out points where vector might be None due to earlier embedding failure
    valid_points_data = [p for p in points_to_upsert if p.get('vector') is not None]
    if not valid_points_data:
        logger.warning("No valid points with vectors to upsert after filtering.")
        return
    try:
        qdrant_points_structs = [
            models.PointStruct(id=p['id'], vector=p['vector'], payload=p['payload'])
            for p in valid_points_data
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=qdrant_points_structs)
        logger.info(f"Upserted {len(qdrant_points_structs)} points to Qdrant collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logger.error(f"Qdrant upsert error: {e}")
        st.error(f"Failed to upsert data to Qdrant: {e}")

@st.cache_data(ttl=3600) # Cache for an hour
def extract_answer_span(query: str, context_chunk: str, model_name: str) -> str:
    try:
        prompt = (
            f"User Query: \"{query}\"\n\n"
            f"Text Chunk:\n\"\"\"{context_chunk}\"\"\"\n\n"
            f"Instruction: From the 'Text Chunk' provided, extract the single, most direct and concise "
            f"sentence or short passage that answers the 'User Query'. The extraction must be verbatim. "
            f"If no part directly answers, extract the most relevant single sentence. "
            f"If the entire chunk is the best short answer, return the entire chunk. "
            f"Do not add any explanation, just the extracted text.\n\nExtracted Answer:"
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.0
        )
        extracted_text = response.choices[0].message.content.strip()
        # Remove potential quotes LLM might add
        if (extracted_text.startswith('"') and extracted_text.endswith('"')) or \
           (extracted_text.startswith("'") and extracted_text.endswith("'")):
            extracted_text = extracted_text[1:-1]
        
        if not extracted_text.strip(): # If empty or just whitespace
            logger.info(f"Answer span extraction returned empty for query '{query}'. Fallback to full chunk.")
            return context_chunk
        return extracted_text
    except OpenAIError as e:
        logger.error(f"OpenAI API error during answer extraction (model: {model_name}): {e}")
        return context_chunk # Fallback
    except Exception as e:
        logger.error(f"Unexpected error during answer extraction (model: {model_name}): {e}")
        return context_chunk # Fallback

@st.cache_data(ttl=120) # Cache for 2 minutes
def get_processed_transcripts() -> list:
    processed_names = set()
    try:
        offset = None
        # Scroll through the collection to get unique transcript names
        # Fetch in batches to avoid overwhelming memory for very large collections
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=250,  # Adjust batch size as needed
                offset=offset,
                with_payload=["transcript_name"], # Only fetch the necessary payload field
                with_vectors=False # No need for vectors here
            )
            for point in points:
                if point.payload and 'transcript_name' in point.payload:
                    processed_names.add(point.payload['transcript_name'])
            
            if not next_offset or not points: # No more points or empty batch
                break
            offset = next_offset
            if len(processed_names) > 1000: # Safety break for extremely large number of unique transcripts
                logger.warning("Reached safety limit for fetching processed transcript names.")
                break
        return sorted(list(processed_names))
    except Exception as e:
        logger.error(f"Error fetching processed transcript list from Qdrant: {e}")
        # st.sidebar.error("Could not fetch transcript list.", icon="⚠️") # UI in main thread
        return []

# === MAIN STREAMLIT UI ===
st.set_page_config(page_title="Gurudev Satsang Search", layout="wide", initial_sidebar_state="expanded")
st.title("♻️ Gurudev's Words – Satsang Archive Search")

# --- Sidebar for Ingestion and Info ---
with st.sidebar:
    st.header("⚙️ Ingestion Controls")
    uploaded_file = st.file_uploader("Upload .srt transcript", type=["srt"])

    # Display list of processed transcripts
    st.subheader("📚 Processed Transcripts")
    processed_transcript_list = get_processed_transcripts()
    if not processed_transcript_list and "qdrant_error_fetching_list" not in st.session_state : # Check for previous error
        try:
            # Attempt one more time if list is empty, might be initial connection issue
            qdrant.get_collection(COLLECTION_NAME) # Test connection again lightly
        except Exception as e:
            st.caption(f"Error connecting to Qdrant to fetch list: {e}")
            st.session_state.qdrant_error_fetching_list = True


    if processed_transcript_list:
        with st.expander("Show/Hide List", expanded=False):
            for item_name in processed_transcript_list:
                st.caption(f"• {item_name}")
    elif "qdrant_error_fetching_list" in st.session_state:
         st.caption("Could not fetch list due to Qdrant connection issue.")
    else:
        st.caption("No transcripts processed yet or unable to fetch list.")


    # Transcript Name Input
    # Auto-fill from filename, allow user to override
    default_transcript_name = ""
    if uploaded_file:
        default_transcript_name = Path(uploaded_file.name).stem
        # If new file uploaded, update session state for input field
        if default_transcript_name != st.session_state.get('last_uploaded_filename_stem_for_input', ''):
            st.session_state.current_transcript_name_input = default_transcript_name
            st.session_state.last_uploaded_filename_stem_for_input = default_transcript_name

    # Initialize session state for transcript name if not present
    if 'current_transcript_name_input' not in st.session_state:
        st.session_state.current_transcript_name_input = default_transcript_name
    
    user_entered_transcript_name = st.text_input(
        "Transcript Name (auto-filled from filename)",
        value=st.session_state.current_transcript_name_input,
        key="transcript_name_user_input_field"
    )
    # Update session state if user manually types
    if user_entered_transcript_name != st.session_state.current_transcript_name_input:
        st.session_state.current_transcript_name_input = user_entered_transcript_name
    
    final_transcript_name_for_processing = st.session_state.current_transcript_name_input


    # Ingestion Parameters
    do_tag_entities_globally = st.checkbox("Enable Claude Entity Tagging (Slows Processing)", value=False, key="entity_tag_toggle",
                                     help=f"Uses {ENTITY_MODEL} to tag people, places, etc. Requires CLAUDE_API_KEY.")
    if do_tag_entities_globally and not CLAUDE_API_KEY:
        st.warning("CLAUDE_API_KEY not set. Entity tagging will be skipped.", icon="⚠️")

    chunk_size_words_input = st.number_input("Chunk size (target words)", min_value=50, max_value=2000, value=400, step=50, key="chunk_size_input")
    overlap_words_input = st.number_input("Chunk overlap (target words)", min_value=0, max_value=1000, value=75, step=25, key="overlap_input")

    if overlap_words_input >= chunk_size_words_input and chunk_size_words_input > 0:
        st.warning("Overlap should generally be less than chunk size for effective chunking.", icon="⚠️")

    process_button_clicked = st.button(
        "Process Transcript",
        disabled=not uploaded_file or not final_transcript_name_for_processing.strip(),
        key="process_srt_button"
    )

# --- Processing Logic ---
if process_button_clicked:
    # This block executes when the "Process Transcript" button is pressed
    st.session_state.qdrant_error_fetching_list = False # Reset error flag on new processing attempt

    with st.status(f"Processing '{final_transcript_name_for_processing}'...", expanded=True) as status_container:
        st.write("Checking if transcript already exists...")
        logger.info(f"Processing request for transcript: '{final_transcript_name_for_processing}'")
        already_exists_in_qdrant = False
        try:
            # Check if any point with this transcript_name already exists
            scroll_response, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="transcript_name", match=models.MatchValue(value=final_transcript_name_for_processing))]
                ),
                limit=1,
                with_payload=False, # Don't need payload for this check
                with_vectors=False
            )
            already_exists_in_qdrant = bool(scroll_response) # True if list of points is not empty
        except Exception as e:
            status_container.update(label=f"Error checking Qdrant for existing transcript: {e}", state="error")
            logger.error(f"Error checking Qdrant for existing transcript '{final_transcript_name_for_processing}': {e}")
            st.stop() # Stop further processing on Qdrant error

        if already_exists_in_qdrant:
            status_container.update(
                label=f"Transcript '{final_transcript_name_for_processing}' may already exist. Re-processing might create duplicates if content is identical but IDs differ, or overwrite if IDs are not unique per content. Consider deleting old entries or using unique names.",
                state="warning"
            )
            # st.stop() # Or allow reprocessing with a warning
        
        temp_srt_file_path = ""
        try:
            st.write(f"Saving temporary SRT file for '{uploaded_file.name}'...")
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".srt", delete=False) as temp_file_obj:
                temp_file_obj.write(uploaded_file.getvalue())
                temp_srt_file_path = temp_file_obj.name
            
            st.write("Parsing SRT content...")
            parsed_subs = parse_srt_file(temp_srt_file_path)
            if not parsed_subs:
                status_container.update(label="SRT file is empty or could not be parsed.", state="error")
                st.stop()
            
            st.write(f"Chunking transcript into segments (target size: {chunk_size_words_input} words, overlap: {overlap_words_input} words)...")
            transcript_chunks = srt_to_chunks(parsed_subs, chunk_size_words_input, overlap_words_input)
            total_chunks_created = len(transcript_chunks)
            if total_chunks_created == 0:
                status_container.update(label="No chunks were created from the SRT. Check content and parameters.", state="error")
                st.stop()
            st.write(f"{total_chunks_created} text chunks created for '{final_transcript_name_for_processing}'.")

            # Batch processing for embeddings and Qdrant upsert
            # Use a new progress bar for this section
            embedding_progress = st.progress(0, text="Embedding and Storing Chunks...")
            
            all_points_for_qdrant = [] # Accumulate all points for potentially one large upsert or batched upserts

            for i in range(0, total_chunks_created, EMBED_BATCH_SIZE):
                current_batch_of_chunks = transcript_chunks[i : i + EMBED_BATCH_SIZE]
                current_batch_texts_list = [c['text'] for c in current_batch_of_chunks]
                
                logger.info(f"Generating embeddings for batch starting at chunk index {i}...")
                batch_vectors = batch_get_embeddings(current_batch_texts_list)

                points_in_this_batch_for_qdrant = []
                for chunk_idx_in_batch, (chunk_data, vector_embedding) in enumerate(zip(current_batch_of_chunks, batch_vectors)):
                    actual_chunk_index = i + chunk_idx_in_batch
                    if vector_embedding is None:
                        logger.warning(f"Skipping chunk {actual_chunk_index} due to missing embedding: {chunk_data['text'][:50]}...")
                        embedding_progress.progress(
                            (actual_chunk_index + 1) / total_chunks_created,
                            text=f"Embedding & Storing: {actual_chunk_index + 1}/{total_chunks_created} (Skipped one)"
                        )
                        continue

                    # Prepare metadata payload for Qdrant
                    chunk_payload = {
                        "transcript_name": final_transcript_name_for_processing,
                        "timestamp": chunk_data['timestamp'],
                        "original_text": chunk_data['text']
                        # Add fine-tuned bio extractions here later if FINE_TUNED_BIO_MODEL_ID is set
                    }
                    
                    if do_tag_entities_globally and CLAUDE_API_KEY:
                        logger.info(f"Tagging entities for chunk {actual_chunk_index}: {chunk_data['text'][:50]}...")
                        entities = tag_entities_with_claude(chunk_data['text'])
                        if entities:
                            chunk_payload['entities'] = entities
                    
                    points_in_this_batch_for_qdrant.append({
                        'id': str(uuid.uuid4()), # Unique ID for each chunk point
                        'vector': vector_embedding,
                        'payload': chunk_payload 
                    })
                    embedding_progress.progress(
                        (actual_chunk_index + 1) / total_chunks_created,
                        text=f"Embedding & Storing: {actual_chunk_index + 1}/{total_chunks_created}"
                    )
                
                if points_in_this_batch_for_qdrant:
                    all_points_for_qdrant.extend(points_in_this_batch_for_qdrant)

            if all_points_for_qdrant:
                st.write(f"Upserting {len(all_points_for_qdrant)} points to Qdrant...")
                upsert_to_qdrant(all_points_for_qdrant) # Upsert all collected points
            
            status_container.update(label=f"Transcript '{final_transcript_name_for_processing}' processed and stored successfully!", state="success")
            logger.info(f"Successfully processed and stored transcript: {final_transcript_name_for_processing}")
            get_processed_transcripts.clear() # Clear cache to refresh list in sidebar

        except Exception as e:
            status_container.update(label=f"An error occurred during processing: {e}", state="error")
            logger.error(f"Error during transcript processing for '{final_transcript_name_for_processing}': {e}", exc_info=True)
        finally:
            if temp_srt_file_path and Path(temp_srt_file_path).exists():
                try:
                    Path(temp_srt_file_path).unlink()
                    logger.info(f"Deleted temporary SRT file: {temp_srt_file_path}")
                except OSError as e_del:
                    logger.error(f"Error deleting temporary file {temp_srt_file_path}: {e_del}")

# --- Search UI ---
st.markdown("---")
st.header("🔍 Semantic Search & Analysis")
search_query_input = st.text_input("Enter your search query:", key="main_search_query_input")

# Search Options Columns
col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1:
    use_llm_reranking = st.checkbox("Enable LLM Reranking", value=True, key="rerank_toggle_checkbox",
                                 help=f"Uses {RERANK_MODEL} to reorder results based on query and custom instructions.")
with col_opt2:
    do_pinpoint_answer_extraction = st.checkbox("Pinpoint Answer Snippet", value=True, key="pinpoint_answer_checkbox",
                                          help=f"Uses {ANSWER_EXTRACTION_MODEL} to identify the most relevant part of each retrieved chunk. Can be slower.")
with col_opt3:
    initial_search_results_limit = st.slider("Initial results to retrieve:", min_value=3, max_value=30, value=5, key="search_results_limit_slider",
                                          help="Number of chunks to fetch from Qdrant before reranking/pinpointing.")

custom_reranking_instructions_input = ""
if use_llm_reranking:
    custom_reranking_instructions_input = st.text_area(
        "Custom Reranking Instructions (Optional):",
        placeholder="e.g., 'Prioritize practical advice.' or 'Focus on answers explaining a specific concept.'",
        key="custom_rerank_instructions_input_area",
        height=100
    )

# --- Search and Display Results ---
if search_query_input:
    with st.spinner("Searching and analyzing results..."):
        # 1. Get Query Embedding
        try:
            query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=[search_query_input])
            query_vector_for_search = query_embedding_response.data[0].embedding
        except OpenAIError as e:
            st.error(f"Failed to get embedding for your query: {e}")
            logger.error(f"OpenAI API error getting query embedding: {e}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred while preparing your query: {e}")
            logger.error(f"Unexpected error getting query embedding: {e}")
            st.stop()

        # 2. Search Qdrant
        try:
            logger.info(f"Searching Qdrant for query: '{search_query_input}' with limit {initial_search_results_limit}")
            qdrant_search_hits = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector_for_search,
                limit=initial_search_results_limit,
                with_payload=True # Ensure we get the payload with original_text, etc.
            )
        except Exception as e:
            st.error(f"Error searching Qdrant: {e}")
            logger.error(f"Qdrant search error: {e}")
            st.stop()
            
        retrieved_results_payloads = [hit.payload for hit in qdrant_search_hits if hit.payload]

        if not retrieved_results_payloads:
            st.info("No results found for your query in the processed transcripts.")
            st.stop()

        # 3. Optional LLM Reranking
        if use_llm_reranking and len(retrieved_results_payloads) > 1:
            with st.spinner(f"Reranking results with {RERANK_MODEL}... This may take a moment."):
                logger.info(f"Attempting to rerank {len(retrieved_results_payloads)} results with {RERANK_MODEL}.")
                excerpts_for_llm_rerank = []
                for i, res_payload in enumerate(retrieved_results_payloads):
                    excerpts_for_llm_rerank.append(f"[{i+1}] {res_payload.get('original_text', 'Error: Text not found in payload')}")
                
                rerank_instruction_prefix = "Rerank the following text excerpts based on their relevance to the user's query."
                if custom_reranking_instructions_input:
                    rerank_instruction_prefix += f" Please pay special attention to the following criteria: {custom_reranking_instructions_input}."
                
                llm_rerank_prompt = (
                    f"{rerank_instruction_prefix}\n\n"
                    f"User's query: \"{search_query_input}\"\n\n"
                    f"Excerpts (each prefixed with an index in brackets, e.g., [1]):\n"
                    f"{chr(10).join(excerpts_for_llm_rerank)}\n\n"
                    f"Return *only* a comma-separated list of the original indices (e.g., '3,1,2') of these excerpts, "
                    f"reranked from most relevant to least relevant based on the query and any specific criteria provided. "
                    f"Do not include any other text, explanation, or apologies."
                )
                
                try:
                    rerank_response = client.chat.completions.create(
                        model=RERANK_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an expert assistant that accurately reranks search results based on user queries and specific instructions."},
                            {"role": "user", "content": llm_rerank_prompt}
                        ],
                        max_tokens=200, # Should be enough for a list of ~30 numbers
                        temperature=0.0 # For deterministic ranking
                    )
                    reranked_indices_str_output = rerank_response.choices[0].message.content.strip()
                    logger.info(f"Reranker LLM output: '{reranked_indices_str_output}'")

                    parsed_reranked_indices = []
                    seen_indices_for_rerank = set()
                    if reranked_indices_str_output:
                        # Find all numbers, attempt to convert, and filter
                        raw_indices_from_llm = re.findall(r'\d+', reranked_indices_str_output)
                        for s_idx in raw_indices_from_llm:
                            try:
                                val = int(s_idx) - 1  # Adjust to 0-based index
                                if 0 <= val < len(retrieved_results_payloads) and val not in seen_indices_for_rerank:
                                    parsed_reranked_indices.append(val)
                                    seen_indices_for_rerank.add(val)
                            except ValueError:
                                logger.warning(f"Reranker LLM returned a non-integer index part: '{s_idx}'")
                    
                    if len(parsed_reranked_indices) > 0:
                        # Add any missing original indices to the end to ensure all results are shown
                        original_indices_set = set(range(len(retrieved_results_payloads)))
                        missing_indices_after_rerank = sorted(list(original_indices_set - seen_indices_for_rerank))
                        final_indices_order_after_rerank = parsed_reranked_indices + missing_indices_after_rerank
                        
                        retrieved_results_payloads = [retrieved_results_payloads[i] for i in final_indices_order_after_rerank if 0 <= i < len(retrieved_results_payloads)]
                        st.caption(f"ℹ️ Results reranked by {RERANK_MODEL}.")
                    else:
                        st.warning("Reranking by LLM failed to produce a valid order, or no reordering was suggested. Displaying results in original Qdrant similarity order.", icon="⚠️")
                        logger.warning("Reranking by LLM produced empty/invalid output or no reordering.")

                except OpenAIError as e_rerank:
                    st.warning(f"LLM Reranking API error: {e_rerank}. Displaying results in original Qdrant similarity order.", icon="⚠️")
                    logger.error(f"OpenAI API error during reranking: {e_rerank}")
                except Exception as e_rerank_general:
                    st.warning(f"An unexpected error occurred during LLM reranking: {e_rerank_general}. Displaying results in original Qdrant similarity order.", icon="⚠️")
                    logger.error(f"Unexpected error during reranking: {e_rerank_general}")
        
        # 4. Display Results
        st.subheader(f"Search Results for: \"{search_query_input}\"")
        if not retrieved_results_payloads: # Should have been caught earlier, but good check
            st.info("No relevant information found.")
        else:
            for i, result_payload_data in enumerate(retrieved_results_payloads, 1):
                if not isinstance(result_payload_data, dict):
                    st.error(f"Result {i} has an invalid payload format.")
                    logger.error(f"Invalid payload for result {i}: {result_payload_data}")
                    continue

                full_chunk_text_content = result_payload_data.get('original_text', 'Error: Text not found in payload')
                
                expander_title = f"Result {i} @ {result_payload_data.get('timestamp', 'N/A')} | Transcript: {result_payload_data.get('transcript_name', 'N/A')}"
                with st.expander(expander_title, expanded=(i==1)): # Expand first result by default
                    # 4a. Optional Pinpointed Answer Extraction
                    if do_pinpoint_answer_extraction and full_chunk_text_content != 'Error: Text not found in payload':
                        with st.spinner(f"Extracting pinpointed answer for result {i}..."):
                            pinpointed_answer_span = extract_answer_span(search_query_input, full_chunk_text_content, ANSWER_EXTRACTION_MODEL)
                        
                        # Display if span is meaningful and different from full chunk
                        if pinpointed_answer_span and \
                           pinpointed_answer_span.strip() and \
                           pinpointed_answer_span.strip().lower() != full_chunk_text_content.strip().lower() and \
                           len(pinpointed_answer_span.strip()) < len(full_chunk_text_content.strip()) : # Ensure it's a sub-span
                            st.markdown(f"🎯 **Pinpointed Answer Snippet:**")
                            st.markdown(f"> *{pinpointed_answer_span.strip()}*")
                            st.markdown("--- \n**Full Context Chunk:**")
                        elif pinpointed_answer_span and pinpointed_answer_span.strip().lower() == full_chunk_text_content.strip().lower():
                             st.caption(f"(The full chunk was identified as the most direct answer snippet by {ANSWER_EXTRACTION_MODEL})")
                        # else: No distinct answer span found or it was empty, just show full chunk below

                    # 4b. Display Full Chunk Text
                    st.markdown(f"{full_chunk_text_content}") # Removed blockquote for less indentation
                    
                    # 4c. Display Entities if present
                    entities_data = result_payload_data.get('entities')
                    if entities_data and isinstance(entities_data, dict):
                        entity_parts = []
                        if entities_data.get('people'): entity_parts.append(f"**People:** {', '.join(entities_data['people'])}")
                        if entities_data.get('places'): entity_parts.append(f"**Places:** {', '.join(entities_data['places'])}")
                        if 'self_references' in entities_data: entity_parts.append(f"**Self-ref:** {entities_data.get('self_references', 'N/A')}")
                        if entity_parts:
                            st.markdown("---")
                            st.markdown(" ".join(entity_parts), unsafe_allow_html=True) # Using markdown for bold