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
- Downloadable Excel of Search Results with Entities
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
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv, find_dotenv
import tempfile
import logging
import pandas as pd # For Excel export
from io import BytesIO # For Excel export in memory

# === CONFIG ===
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gurudev_satsangs")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 20))

ENTITY_MODEL = os.getenv("ENTITY_MODEL", "claude-3-opus-20240229") # Used for entities
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4-turbo-preview")
ANSWER_EXTRACTION_MODEL = os.getenv("ANSWER_EXTRACTION_MODEL", "gpt-3.5-turbo")

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
    qdrant_url = QDRANT_HOST
    if "://" not in qdrant_url:
        qdrant_url = f"https://{QDRANT_HOST}:{QDRANT_PORT}"

    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
    collections_response = qdrant.get_collections()
    logger.info(f"Successfully connected to Qdrant. Available collections: {collections_response.collections}")
except Exception as e:
    st.error(f"Qdrant connection error: {e}. Check QDRANT_HOST, QDRANT_PORT, and QDRANT_API_KEY.")
    logger.error(f"Qdrant connection error: {e}")
    st.stop()

try:
    collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
    if collection_info.config.params.vectors.size != EMBEDDING_DIM or \
       collection_info.config.params.vectors.distance != models.Distance.COSINE:
        logger.warning(f"Collection '{COLLECTION_NAME}' vector config differs. Recreating.")
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
        )
        logger.info(f"Recreated Qdrant collection: {COLLECTION_NAME}")
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="transcript_name",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        logger.info(f"Created payload index on 'transcript_name' for collection '{COLLECTION_NAME}'.")
    else:
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="transcript_name",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Ensured payload index on 'transcript_name' (KEYWORD) exists for '{COLLECTION_NAME}'.")
        except Exception as e_index_create:
            logger.warning(f"Could not definitively create payload index on 'transcript_name' (may already exist or other issue): {e_index_create}. If it's an 'already exists' info, it's often okay.")
except Exception as e_col:
    is_not_found_error = "Not found" in str(e_col) or \
                         "404" in str(e_col) or \
                         (hasattr(e_col, 'status_code') and e_col.status_code == 404) or \
                         "status_code=NOT_FOUND" in str(e_col)
    if is_not_found_error:
        logger.info(f"Collection '{COLLECTION_NAME}' not found. Creating collection and index.")
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="transcript_name",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        logger.info(f"Created payload index on 'transcript_name' for new collection '{COLLECTION_NAME}'.")
    else:
        st.error(f"Error accessing or creating Qdrant collection '{COLLECTION_NAME}': {e_col}")
        logger.error(f"Error accessing or creating Qdrant collection '{COLLECTION_NAME}': {e_col}")
        st.stop()

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
            for i in range(len(buffer) - 1, 0, -1):
                retained_words_for_overlap += len(buffer[i].content.split())
                subs_in_overlap_count += 1
                if retained_words_for_overlap >= overlap_words:
                    break
            subs_to_advance_from_chunk_start = max(1, len(buffer) - subs_in_overlap_count)
            next_start_idx = current_sub_idx + subs_to_advance_from_chunk_start
            if next_start_idx <= current_sub_idx :
                current_sub_idx = temp_idx
            else:
                current_sub_idx = min(next_start_idx, len(subs))
    logger.info(f"Created {len(chunks)} chunks from SRT. Target size: {chunk_size_words} words, overlap: {overlap_words} words.")
    return chunks

def tag_entities_with_claude(text: str) -> dict:
    if not CLAUDE_API_KEY:
        return {}
    headers = {"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    prompt = (
        f"Analyze the following text and extract key entities. "
        f"Respond *only* with a single JSON object containing 'people' (list of strings), "
        f"'places' (list of strings), and 'self_references' (boolean, true if the speaker refers to themselves, e.g., 'I', 'me', 'my'). "
        f"Example JSON: {{\"people\": [\"Krishna\", \"Arjuna\"], \"places\": [\"Kurukshetra\"], \"self_references\": true}}\n"
        f"If no entities of a type are found, use an empty list. If no self-references, use false.\n"
        f"Text:\n{text}\nJSON Output:"
    )
    body = {"model": ENTITY_MODEL, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    try:
        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(CLAUDE_API_URL, headers=headers, json=body)
            response.raise_for_status()
        data = response.json()
        if data.get('content') and data['content'][0].get('type') == 'text':
            raw_text_response = data['content'][0]['text']
            json_str = None
            json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_text_response, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1)
            else:
                start_brace = raw_text_response.find('{')
                end_brace = raw_text_response.rfind('}')
                if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                    json_str = raw_text_response[start_brace : end_brace + 1]
            
            if not json_str:
                logger.error(f"Claude Entity: No JSON object found in response: {raw_text_response[:200]}")
                return {}
            try:
                entities = json.loads(json_str)
                return {
                    'people': entities.get('people', []) if isinstance(entities.get('people'), list) else [],
                    'places': entities.get('places', []) if isinstance(entities.get('places'), list) else [],
                    'self_references': entities.get('self_references', False) if isinstance(entities.get('self_references'), bool) else False
                }
            except json.JSONDecodeError as e:
                logger.error(f"Claude Entity JSON parse error: {e}. Response: {json_str[:200]}"); return {}
        return {}
    except httpx.HTTPStatusError as e:
        logger.error(f"Claude Entity API HTTP error: {e.response.status_code} - {e.response.text}"); return {}
    except Exception as e:
        logger.error(f"Claude Entity API/tagging error: {e}"); return {}

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
    valid_points_data = [p for p in points_to_upsert if p.get('vector') is not None]
    if not valid_points_data:
        logger.warning("No valid points with vectors to upsert after filtering.")
        return
    try:
        qdrant_points_structs = [
            models.PointStruct(id=p['id'], vector=p['vector'], payload=p['payload'])
            for p in valid_points_data
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=qdrant_points_structs, wait=True)
        logger.info(f"Upserted {len(qdrant_points_structs)} points to Qdrant collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logger.error(f"Qdrant upsert error: {e}")
        st.error(f"Failed to upsert data to Qdrant: {e}")

@st.cache_data(ttl=3600)
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
        if (extracted_text.startswith('"') and extracted_text.endswith('"')) or \
           (extracted_text.startswith("'") and extracted_text.endswith("'")):
            extracted_text = extracted_text[1:-1]
        
        if not extracted_text.strip():
            logger.info(f"Answer span extraction returned empty for query '{query}'. Fallback to full chunk.")
            return context_chunk
        return extracted_text
    except OpenAIError as e:
        logger.error(f"OpenAI API error during answer extraction (model: {model_name}): {e}")
        return context_chunk 
    except Exception as e:
        logger.error(f"Unexpected error during answer extraction (model: {model_name}): {e}")
        return context_chunk

@st.cache_data(ttl=120)
def get_processed_transcripts() -> list:
    processed_names = set()
    try:
        offset = None
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=250, 
                offset=offset,
                with_payload=["transcript_name"],
                with_vectors=False
            )
            for point in points:
                if point.payload and 'transcript_name' in point.payload:
                    processed_names.add(point.payload['transcript_name'])
            
            if not next_offset or not points:
                break
            offset = next_offset
            if len(processed_names) > 1000:
                logger.warning("Reached safety limit for fetching processed transcript names (1000 unique names).")
                break
        return sorted(list(processed_names))
    except Exception as e:
        logger.error(f"Error fetching processed transcript list from Qdrant: {e}")
        return []

def convert_results_to_excel(results_payloads: list) -> bytes:
    """Converts a list of Qdrant result payloads to an Excel file in memory."""
    data_for_df = []
    for payload in results_payloads:
        if not isinstance(payload, dict):
            logger.warning(f"Skipping invalid payload for Excel export: {payload}")
            continue

        transcript_name = payload.get('transcript_name', 'N/A')
        timestamp = payload.get('timestamp', 'N/A')
        chunk_text = payload.get('original_text', '')
        
        entities_data = payload.get('entities', {})
        people_list = []
        places_list = []

        if isinstance(entities_data, dict):
            people_list = entities_data.get('people', [])
            places_list = entities_data.get('places', [])
        
        # Ensure they are lists and join them, handle non-list gracefully
        people_str = ", ".join(people_list) if isinstance(people_list, list) else ""
        places_str = ", ".join(places_list) if isinstance(places_list, list) else ""

        data_for_df.append({
            'Transcript Name': transcript_name,
            'Timestamp': timestamp,
            'Chunk Text': chunk_text,
            'People': people_str,
            'Places': places_str
        })

    df = pd.DataFrame(data_for_df)
    
    output = BytesIO()
    # Use a context manager for ExcelWriter to ensure it's closed properly
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Satsang Chunks')
    # The stream is written to by ExcelWriter, get its value after writer is closed
    processed_data = output.getvalue()
    return processed_data

# === MAIN STREAMLIT UI ===
st.set_page_config(page_title="Gurudev Satsang Search", layout="wide", initial_sidebar_state="expanded")
st.title("♻️ Gurudev's Words – Satsang Archive Search")

# --- Sidebar for Ingestion and Info ---
with st.sidebar:
    st.header("⚙️ Ingestion Controls")
    uploaded_file = st.file_uploader("Upload .srt transcript", type=["srt"])

    st.subheader("📚 Processed Transcripts")
    processed_transcript_list = get_processed_transcripts()
    if not processed_transcript_list and "qdrant_error_fetching_list" not in st.session_state :
        try:
            qdrant.get_collection(COLLECTION_NAME) 
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

    default_transcript_name = ""
    if uploaded_file:
        default_transcript_name = Path(uploaded_file.name).stem
        if default_transcript_name != st.session_state.get('last_uploaded_filename_stem_for_input', ''):
            st.session_state.current_transcript_name_input = default_transcript_name
            st.session_state.last_uploaded_filename_stem_for_input = default_transcript_name

    if 'current_transcript_name_input' not in st.session_state:
        st.session_state.current_transcript_name_input = default_transcript_name
    
    user_entered_transcript_name = st.text_input(
        "Transcript Name (auto-filled from filename)",
        value=st.session_state.current_transcript_name_input,
        key="transcript_name_user_input_field"
    )
    if user_entered_transcript_name != st.session_state.current_transcript_name_input:
        st.session_state.current_transcript_name_input = user_entered_transcript_name
    
    final_transcript_name_for_processing = st.session_state.current_transcript_name_input

    do_tag_entities_globally = st.checkbox(
        "Enable Claude Entity Tagging", value=False, key="entity_tag_toggle",
        help=f"Uses {ENTITY_MODEL} to tag people, places, etc. Requires CLAUDE_API_KEY."
    )
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
    st.session_state.qdrant_error_fetching_list = False 

    with st.status(f"Processing '{final_transcript_name_for_processing}'...", expanded=True) as status_container:
        st.write("Checking if transcript already exists...")
        logger.info(f"Processing request for transcript: '{final_transcript_name_for_processing}'")
        already_exists_in_qdrant = False
        try:
            scroll_response, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="transcript_name", match=models.MatchValue(value=final_transcript_name_for_processing))]
                ),
                limit=1, with_payload=False, with_vectors=False
            )
            already_exists_in_qdrant = bool(scroll_response)
        except Exception as e:
            status_container.update(label=f"Error checking Qdrant for existing transcript: {e}", state="error")
            logger.error(f"Error checking Qdrant for existing transcript '{final_transcript_name_for_processing}': {e}")
            st.stop()

        if already_exists_in_qdrant:
            st.write(
                f"Warning: Transcript named '{final_transcript_name_for_processing}' may already have entries in the database. "
                f"Continuing will add new chunks which might be duplicates if the content is the same. "
                f"Consider using a unique name if this is a different version or a new transcript."
            )
            logger.warning(f"Transcript '{final_transcript_name_for_processing}' may already exist. Continuing with processing.")
        
        temp_srt_file_path = ""
        try:
            st.write(f"Saving temporary SRT file for '{uploaded_file.name}'...")
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".srt", delete=False) as temp_file_obj:
                temp_file_obj.write(uploaded_file.getvalue())
                temp_srt_file_path = temp_file_obj.name
            
            st.write("Parsing SRT content...")
            parsed_subs = parse_srt_file(temp_srt_file_path)
            if not parsed_subs:
                status_container.update(label="SRT file is empty or could not be parsed.", state="error"); st.stop()
            
            st.write(f"Chunking transcript into segments (target size: {chunk_size_words_input} words, overlap: {overlap_words_input} words)...")
            transcript_chunks = srt_to_chunks(parsed_subs, chunk_size_words_input, overlap_words_input)
            total_chunks_created = len(transcript_chunks)
            if total_chunks_created == 0:
                status_container.update(label="No chunks were created from the SRT. Check content and parameters.", state="error"); st.stop()
            st.write(f"{total_chunks_created} text chunks created for '{final_transcript_name_for_processing}'.")

            embedding_progress = st.progress(0.0, text="Preparing to process chunks...")
            all_points_for_qdrant = []

            for i in range(0, total_chunks_created, EMBED_BATCH_SIZE):
                current_batch_of_chunks = transcript_chunks[i : i + EMBED_BATCH_SIZE]
                current_batch_texts_list = [c['text'] for c in current_batch_of_chunks]
                
                batch_start_index_display = i + 1
                batch_end_index_display = min(i + EMBED_BATCH_SIZE, total_chunks_created)
                embedding_progress.progress(
                    (i / total_chunks_created), 
                    text=f"Embedding batch {batch_start_index_display}-{batch_end_index_display}/{total_chunks_created}..."
                )
                
                logger.info(f"Generating embeddings for batch of {len(current_batch_texts_list)} chunks (starting original index {i})...")
                batch_vectors = batch_get_embeddings(current_batch_texts_list)

                points_in_this_batch_for_qdrant = []
                for chunk_idx_in_batch, (chunk_data, vector_embedding) in enumerate(zip(current_batch_of_chunks, batch_vectors)):
                    actual_chunk_index = i + chunk_idx_in_batch
                    
                    progress_value = (actual_chunk_index + 0.5) / total_chunks_created
                    progress_text = f"Processing chunk {actual_chunk_index + 1}/{total_chunks_created}"
                    embedding_progress.progress(progress_value, text=progress_text)

                    if vector_embedding is None:
                        logger.warning(f"Skipping chunk {actual_chunk_index} from '{final_transcript_name_for_processing}' due to missing embedding: {chunk_data['text'][:50]}...")
                        continue

                    chunk_payload = {
                        "transcript_name": final_transcript_name_for_processing,
                        "timestamp": chunk_data['timestamp'],
                        "original_text": chunk_data['text']
                    }
                    
                    if do_tag_entities_globally and CLAUDE_API_KEY:
                        embedding_progress.progress(progress_value, text=f"{progress_text} (Entity Tagging...)")
                        logger.info(f"Tagging entities for chunk {actual_chunk_index} of '{final_transcript_name_for_processing}': {chunk_data['text'][:50]}...")
                        entities = tag_entities_with_claude(chunk_data['text'])
                        if entities: chunk_payload['entities'] = entities
                    
                    points_in_this_batch_for_qdrant.append({
                        'id': str(uuid.uuid4()), 
                        'vector': vector_embedding,
                        'payload': chunk_payload 
                    })
                
                if points_in_this_batch_for_qdrant:
                    all_points_for_qdrant.extend(points_in_this_batch_for_qdrant)
                
                embedding_progress.progress(
                    ((i + len(current_batch_of_chunks)) / total_chunks_created), 
                    text=f"Batch {batch_start_index_display}-{batch_end_index_display}/{total_chunks_created} processed."
                )

            if all_points_for_qdrant:
                embedding_progress.progress(0.95, text=f"Upserting {len(all_points_for_qdrant)} points to Qdrant...")
                upsert_to_qdrant(all_points_for_qdrant)
            
            embedding_progress.progress(1.0)
            status_container.update(label=f"Transcript '{final_transcript_name_for_processing}' processed and stored successfully!", state="success")
            logger.info(f"Successfully processed and stored transcript: {final_transcript_name_for_processing}")
            get_processed_transcripts.clear()

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

# Initialize session state for search results if not present
if 'retrieved_results_payloads_for_download' not in st.session_state:
    st.session_state.retrieved_results_payloads_for_download = []


# --- Search and Display Results ---
if search_query_input:
    with st.spinner("Searching and analyzing results..."):
        try:
            query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=[search_query_input])
            query_vector_for_search = query_embedding_response.data[0].embedding
        except OpenAIError as e:
            st.error(f"Failed to get embedding for your query: {e}"); logger.error(f"OpenAI API error getting query embedding: {e}"); st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred while preparing your query: {e}"); logger.error(f"Unexpected error getting query embedding: {e}"); st.stop()

        try:
            logger.info(f"Searching Qdrant for query: '{search_query_input}' with limit {initial_search_results_limit}")
            qdrant_search_hits = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector_for_search,
                limit=initial_search_results_limit,
                with_payload=True
            )
        except Exception as e:
            st.error(f"Error searching Qdrant: {e}"); logger.error(f"Qdrant search error: {e}"); st.stop()
            
        retrieved_results_payloads = [hit.payload for hit in qdrant_search_hits if hit.payload]
        st.session_state.retrieved_results_payloads_for_download = retrieved_results_payloads # Store for download

        if not retrieved_results_payloads:
            st.info("No results found for your query in the processed transcripts."); st.stop()

        if use_llm_reranking and len(retrieved_results_payloads) > 1:
            with st.spinner(f"Reranking results with {RERANK_MODEL}... This may take a moment."):
                # (Reranking logic - unchanged from previous correct version)
                logger.info(f"Attempting to rerank {len(retrieved_results_payloads)} results with {RERANK_MODEL}.")
                excerpts_for_llm_rerank = [f"[{i+1}] {res_payload.get('original_text', 'Error: Text not found in payload')}" for i, res_payload in enumerate(retrieved_results_payloads)]
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
                        max_tokens=max(200, len(retrieved_results_payloads) * 6),
                        temperature=0.0
                    )
                    reranked_indices_str_output = rerank_response.choices[0].message.content.strip()
                    logger.info(f"Reranker LLM output: '{reranked_indices_str_output}'")
                    parsed_reranked_indices = []
                    seen_indices_for_rerank = set()
                    if reranked_indices_str_output:
                        raw_indices_from_llm = re.findall(r'\d+', reranked_indices_str_output)
                        for s_idx in raw_indices_from_llm:
                            try:
                                val = int(s_idx) - 1 
                                if 0 <= val < len(retrieved_results_payloads) and val not in seen_indices_for_rerank:
                                    parsed_reranked_indices.append(val)
                                    seen_indices_for_rerank.add(val)
                            except ValueError: logger.warning(f"Reranker LLM returned a non-integer index part: '{s_idx}'")
                    if len(parsed_reranked_indices) > 0:
                        original_indices_set = set(range(len(retrieved_results_payloads)))
                        missing_indices_after_rerank = sorted(list(original_indices_set - seen_indices_for_rerank))
                        final_indices_order_after_rerank = parsed_reranked_indices + missing_indices_after_rerank
                        if len(final_indices_order_after_rerank) == len(retrieved_results_payloads):
                             retrieved_results_payloads = [retrieved_results_payloads[i] for i in final_indices_order_after_rerank if 0 <= i < len(retrieved_results_payloads)]
                             st.session_state.retrieved_results_payloads_for_download = retrieved_results_payloads # Update for download too
                             st.caption(f"ℹ️ Results reranked by {RERANK_MODEL}.")
                        else:
                            st.warning("Reranking by LLM resulted in an inconsistent number of items. Displaying in original Qdrant similarity order.", icon="⚠️")
                            logger.warning("Reranking by LLM produced inconsistent index count.")
                    else:
                        st.warning("Reranking by LLM failed to produce a valid order, or no reordering was suggested. Displaying results in original Qdrant similarity order.", icon="⚠️")
                        logger.warning("Reranking by LLM produced empty/invalid output or no reordering.")
                except OpenAIError as e_rerank:
                    st.warning(f"LLM Reranking API error: {e_rerank}. Displaying results in original Qdrant similarity order.", icon="⚠️")
                    logger.error(f"OpenAI API error during reranking: {e_rerank}")
                except Exception as e_rerank_general:
                    st.warning(f"An unexpected error occurred during LLM reranking: {e_rerank_general}. Displaying results in original Qdrant similarity order.", icon="⚠️")
                    logger.error(f"Unexpected error during reranking: {e_rerank_general}")
        
        st.subheader(f"Search Results for: \"{search_query_input}\"")

        # --- Add Download Button Here ---
        if st.session_state.retrieved_results_payloads_for_download:
            excel_data = convert_results_to_excel(st.session_state.retrieved_results_payloads_for_download)
            st.download_button(
                label="📥 Download Results as Excel",
                data=excel_data,
                file_name="satsang_search_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_button"
            )
        # --- End Download Button ---

        if not retrieved_results_payloads: 
            st.info("No relevant information found.") # Should have been caught earlier, but good for clarity
        else:
            for i, result_payload_data in enumerate(retrieved_results_payloads, 1):
                if not isinstance(result_payload_data, dict):
                    st.error(f"Result {i} has an invalid payload format.")
                    logger.error(f"Invalid payload for result {i}: {result_payload_data}")
                    continue

                full_chunk_text_content = result_payload_data.get('original_text', 'Error: Text not found in payload')
                
                expander_title = f"Result {i} @ {result_payload_data.get('timestamp', 'N/A')} | Transcript: {result_payload_data.get('transcript_name', 'N/A')}"
                with st.expander(expander_title, expanded=(i==1)): 
                    if do_pinpoint_answer_extraction and full_chunk_text_content != 'Error: Text not found in payload':
                        with st.spinner(f"Extracting pinpointed answer for result {i}..."):
                            pinpointed_answer_span = extract_answer_span(search_query_input, full_chunk_text_content, ANSWER_EXTRACTION_MODEL)
                        
                        if pinpointed_answer_span and \
                           pinpointed_answer_span.strip() and \
                           pinpointed_answer_span.strip().lower() != full_chunk_text_content.strip().lower() and \
                           len(pinpointed_answer_span.strip()) < len(full_chunk_text_content.strip()) :
                            st.markdown(f"🎯 **Pinpointed Answer Snippet:**")
                            st.markdown(f"> *{pinpointed_answer_span.strip()}*")
                            st.markdown("--- \n**Full Context Chunk:**")
                        elif pinpointed_answer_span and pinpointed_answer_span.strip().lower() == full_chunk_text_content.strip().lower():
                             st.caption(f"(The full chunk was identified as the most direct answer snippet by {ANSWER_EXTRACTION_MODEL})")

                    st.markdown(f"{full_chunk_text_content}") 
                    
                    entities_data = result_payload_data.get('entities')
                    if entities_data and isinstance(entities_data, dict):
                        entity_parts = []
                        if entities_data.get('people'): entity_parts.append(f"**People:** {', '.join(entities_data['people'])}")
                        if entities_data.get('places'): entity_parts.append(f"**Places:** {', '.join(entities_data['places'])}")
                        if 'self_references' in entities_data and entities_data.get('self_references') is not None:
                             entity_parts.append(f"**Self-ref:** {'Yes' if entities_data['self_references'] else 'No'}")
                        if entity_parts:
                            st.markdown("---")
                            st.markdown(" ".join(entity_parts), unsafe_allow_html=True)
else:
    # If there's no search query, clear the session state for download
    st.session_state.retrieved_results_payloads_for_download = []