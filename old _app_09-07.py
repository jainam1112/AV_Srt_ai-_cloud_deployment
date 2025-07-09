Conversation opened. 1 read message.

Skip to content
Using Gmail with screen readers
in:sent 
3 of 878
prev code

jainam dedhia <jainamdedhia12@gmail.com>
Attachments
Fri, Jun 27, 4:42 PM (12 days ago)
to Kinnari

pfa stable code

--
In sewa,
Jainam Dedhia
 One attachment
  •  Scanned by Gmail
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gurudev Satsang Transcript Processing Pipeline + Streamlit UI
- SRT Upload and Processing (without initial entity tagging)
- Chunking with Overlap
- OpenAI Embeddings
- Qdrant Vector Storage
- Semantic Search
- LLM-based Reranking with Custom User Instructions
- LLM-based Answer Span Extraction
- Displaying a List of Already Processed Transcripts
- SEPARATE POST-PROCESSING STEP FOR CLAUDE ENTITY TAGGING
- NEW: Inspect all chunks and payloads for a selected transcript
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
ENTITY_MODEL = os.getenv("ENTITY_MODEL", "claude-3-opus-20240229") 
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4-turbo-preview")
ANSWER_EXTRACTION_MODEL = os.getenv("ANSWER_EXTRACTION_MODEL", "gpt-3.5-turbo")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === INITIALIZATION ===
if not OPENAI_API_KEY: st.error("OPENAI_API_KEY not found."); st.stop()
try: client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e: st.error(f"OpenAI client init error: {e}"); logger.error(f"OpenAI init error: {e}"); st.stop()

qdrant_client = None 
if QDRANT_API_KEY and QDRANT_HOST:
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, prefer_grpc=True) 
        qdrant_client.get_collections()
        logger.info(f"Successfully connected to Qdrant Cloud at {QDRANT_HOST}")
    except Exception as e:
        st.error(f"Qdrant Cloud connection error: {e}.")
        logger.error(f"Qdrant Cloud connection error: {e}")
        qdrant_client = None 
        st.stop() 
else:
    st.error("QDRANT_HOST and QDRANT_API_KEY must be set for Qdrant Cloud.")
    logger.error("QDRANT_HOST or QDRANT_API_KEY not set.")
    st.stop()

if qdrant_client:
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        if collection_info.config.params.vectors.size != EMBEDDING_DIM or \
           collection_info.config.params.vectors.distance != models.Distance.COSINE:
            logger.warning(f"Collection '{COLLECTION_NAME}' vector config differs. Recreating.")
            qdrant_client.recreate_collection(collection_name=COLLECTION_NAME, vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE))
            logger.info(f"Recreated Qdrant collection: {COLLECTION_NAME}")
            qdrant_client.create_payload_index(collection_name=COLLECTION_NAME, field_name="transcript_name", field_schema=models.PayloadSchemaType.KEYWORD)
            logger.info(f"Created payload index on 'transcript_name' for '{COLLECTION_NAME}'.")
        else: 
            try:
                qdrant_client.create_payload_index(collection_name=COLLECTION_NAME, field_name="transcript_name", field_schema=models.PayloadSchemaType.KEYWORD)
                logger.info(f"Ensured payload index on 'transcript_name' exists for '{COLLECTION_NAME}'.")
            except Exception as e_idx_create: 
                if "already exists" in str(e_idx_create).lower():
                    logger.info(f"Payload index on 'transcript_name' likely already exists for '{COLLECTION_NAME}'.")
                else:
                    logger.warning(f"Issue ensuring payload index on 'transcript_name': {e_idx_create}.")
    except Exception as e_col:
        if "not found" in str(e_col).lower() or ("status_code" in dir(e_col) and e_col.status_code == 404): # Check if status_code attr exists
            logger.info(f"Collection '{COLLECTION_NAME}' not found. Creating collection and index.")
            qdrant_client.recreate_collection(collection_name=COLLECTION_NAME, vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE))
            qdrant_client.create_payload_index(collection_name=COLLECTION_NAME, field_name="transcript_name", field_schema=models.PayloadSchemaType.KEYWORD)
            logger.info(f"Created Qdrant collection and index: {COLLECTION_NAME}")
        else:
            st.error(f"Error accessing/creating Qdrant collection '{COLLECTION_NAME}': {e_col}")
            logger.error(f"Qdrant collection error: {e_col}")
            st.stop()

# === HELPERS ===
def parse_srt_file(file_path: str) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return list(srt.parse(f.read()))
    except Exception as e: logger.error(f"SRT parse error {file_path}: {e}"); st.error(f"SRT parse error: {e}"); return []

def srt_to_chunks(subs: list, chunk_size_words: int, overlap_words: int) -> list:
    chunks, current_sub_idx = [], 0
    if not subs: return []
    while current_sub_idx < len(subs):
        buffer, word_count, temp_idx = [], 0, current_sub_idx
        while temp_idx < len(subs):
            sub = subs[temp_idx]; words_in_sub = len(sub.content.split())
            if not buffer or word_count + words_in_sub <= chunk_size_words + (chunk_size_words * 0.2):
                buffer.append(sub); word_count += words_in_sub
            else: break
            temp_idx += 1
        if not buffer: break
        chunks.append({'text': ' '.join(s.content for s in buffer), 'timestamp': str(buffer[0].start)})
        if overlap_words <= 0 or len(buffer) <= 1: current_sub_idx = temp_idx
        else:
            retained_words, subs_in_overlap = 0, 0
            for i in range(len(buffer) - 1, 0, -1):
                retained_words += len(buffer[i].content.split()); subs_in_overlap += 1
                if retained_words >= overlap_words: break
            advance_by = max(1, len(buffer) - subs_in_overlap)
            next_start = current_sub_idx + advance_by
            current_sub_idx = temp_idx if next_start <= current_sub_idx else min(next_start, len(subs))
    logger.info(f"Created {len(chunks)} chunks."); return chunks

def tag_entities_with_claude(text: str) -> dict:
    if not CLAUDE_API_KEY: logger.warning("CLAUDE_API_KEY not set."); return {} 
    headers = {"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    prompt = (f"Analyze text, extract entities. Respond *only* with JSON: {{'people': [], 'places': [], 'self_references': bool}}.\nText:\n{text}\nJSON Output:")
    body = {"model": ENTITY_MODEL, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}
    try:
        with httpx.Client(timeout=45.0) as http_client: 
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
            except json.JSONDecodeError as e: logger.error(f"Claude JSON parse error: {e}. Resp: {json_str[:100]}"); return {}
        return {}
    except httpx.HTTPStatusError as e: logger.error(f"Claude HTTP error: {e.response.status_code} - {e.response.text}"); return {}
    except Exception as e: logger.error(f"Claude API/tagging error: {e}"); return {}

def batch_get_embeddings(texts: list) -> list:
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([obj.embedding for obj in response.data])
        except OpenAIError as e: logger.error(f"OpenAI embed error: {e}"); st.error(f"OpenAI embed error: {e}"); all_embeddings.extend([None]*len(batch))
        except Exception as e: logger.error(f"Unexpected embed error: {e}"); st.error(f"Unexpected embed error: {e}"); all_embeddings.extend([None]*len(batch))
    return all_embeddings

def upsert_to_qdrant(points_to_upsert: list): 
    if not points_to_upsert or not qdrant_client: return
    valid_points = [p for p in points_to_upsert if p.get('vector') is not None]
    if not valid_points: logger.warning("No valid points for Qdrant upsert."); return
    try:
        structs = [models.PointStruct(id=p['id'], vector=p['vector'], payload=p['payload']) for p in valid_points]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=structs)
        logger.info(f"Upserted {len(structs)} points to Qdrant.")
    except Exception as e: logger.error(f"Qdrant upsert error: {e}"); st.error(f"Qdrant upsert error: {e}")

@st.cache_data(ttl=3600)
def extract_answer_span(query: str, context_chunk: str, model_name: str) -> str:
    try:
        prompt = (f"Query: \"{query}\"\nChunk:\n\"\"\"{context_chunk}\"\"\"\nInstruction: Extract verbatim sentence(s) from Chunk answering Query. If none, extract most relevant. If all, return all. Only extracted text.\nExtracted:")
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=0.0)
        extracted = response.choices[0].message.content.strip()
        if (extracted.startswith('"') and extracted.endswith('"')) or (extracted.startswith("'") and extracted.endswith("'")): extracted = extracted[1:-1]
        return extracted if extracted.strip() else context_chunk
    except Exception as e: logger.error(f"Answer extraction error ({model_name}): {e}"); return context_chunk

@st.cache_data(ttl=120)
def get_processed_transcripts() -> list: 
    if not qdrant_client: return []
    names = set()
    try:
        offset_val = None
        while True:
            points_batch, next_offset_val = qdrant_client.scroll(
                collection_name=COLLECTION_NAME, limit=250, offset=offset_val,
                with_payload=["transcript_name"], with_vectors=False
            )
            for p in points_batch:
                if p.payload and 'transcript_name' in p.payload: names.add(p.payload['transcript_name'])
            if not next_offset_val or not points_batch or len(names) > 1000: break # Safety break
            offset_val = next_offset_val
        return sorted(list(names))
    except Exception as e: logger.error(f"Error fetching transcript list: {e}"); return []

def tag_transcript_entities_post_processing(transcript_name_to_tag: str):
    if not qdrant_client: st.error("Qdrant client not available."); return
    if not CLAUDE_API_KEY: st.warning("CLAUDE_API_KEY not set.", icon="⚠️"); return
    st.info(f"Fetching chunks for '{transcript_name_to_tag}' to tag entities...")
    updates_to_perform = [] 
    try:
        offset_val = None; fetched_count = 0; processed_count = 0
        prog_text = st.empty(); prog_bar = st.progress(0)
        while True:
            prog_text.text(f"Fetching batch for '{transcript_name_to_tag}'...")
            retrieved_points, next_offset_val = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[models.FieldCondition(key="transcript_name", match=models.MatchValue(value=transcript_name_to_tag))]),
                limit=50, offset=offset_val, with_payload=True, with_vectors=False
            )
            if not retrieved_points: break
            fetched_count += len(retrieved_points)
            prog_text.text(f"Fetched {fetched_count} chunks. Preparing tags...")
            for point in retrieved_points:
                processed_count +=1
                if point.payload and 'original_text' in point.payload:
                    if point.payload.get('entities') is not None: 
                        logger.info(f"Point {point.id} already tagged. Skipping."); continue
                    chunk_text = point.payload['original_text']
                    logger.info(f"Tagging entities for point ID {point.id}...")
                    prog_text.text(f"Tagging chunk {processed_count} (ID: ...{str(point.id)[-6:]})...")
                    entities = tag_entities_with_claude(chunk_text)
                    if entities: updates_to_perform.append((point.id, {'entities': entities}))
                else: logger.warning(f"Point ID {point.id} missing 'original_text'. Skipping.")
            if not next_offset_val: break
            offset_val = next_offset_val
        prog_bar.empty(); prog_text.empty()
        if not updates_to_perform: st.success(f"No new chunks needed tagging for '{transcript_name_to_tag}'."); return
        st.write(f"Applying entity tags to {len(updates_to_perform)} chunks in '{transcript_name_to_tag}'...")
        apply_prog = st.progress(0, text="Applying tags to Qdrant...")
        for i, (point_id, entity_payload) in enumerate(updates_to_perform):
            try:
                qdrant_client.set_payload(collection_name=COLLECTION_NAME, payload=entity_payload, points=[point_id], wait=True)
                logger.info(f"Successfully set payload for point {point_id}.")
            except Exception as e_set: logger.error(f"Failed to set payload for {point_id}: {e_set}"); st.warning(f"Failed to update point ...{str(point_id)[-6:]}.", icon="⚠️")
            apply_prog.progress((i + 1) / len(updates_to_perform), text=f"Applying tags: {i + 1}/{len(updates_to_perform)}")
        st.success(f"Entity tagging complete for '{transcript_name_to_tag}'. {len(updates_to_perform)} chunks updated/checked.")
        get_processed_transcripts.clear()
    except Exception as e:
        st.error(f"Error during post-processing entity tagging for '{transcript_name_to_tag}': {e}")
        logger.error(f"Error during post-processing tagging: {e}", exc_info=True)
        if 'prog_bar' in locals() and prog_bar: prog_bar.empty()
        if 'prog_text' in locals() and prog_text: prog_text.empty()

@st.cache_data(ttl=30) 
def get_all_chunks_for_transcript(transcript_name_to_inspect: str) -> list:
    if not qdrant_client or not transcript_name_to_inspect: return []
    all_points_data = []
    try:
        offset_val = None
        while True:
            points_batch, next_offset_val = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[models.FieldCondition(key="transcript_name", match=models.MatchValue(value=transcript_name_to_inspect))]),
                limit=100, offset=offset_val, with_payload=True, with_vectors=False
            )
            if not points_batch: break
            for point in points_batch:
                all_points_data.append({"id": str(point.id), "payload": point.payload})
            if not next_offset_val: break
            offset_val = next_offset_val
        logger.info(f"Retrieved {len(all_points_data)} chunks for inspection from '{transcript_name_to_inspect}'.")
        return all_points_data
    except Exception as e:
        st.error(f"Error fetching chunks for inspection from '{transcript_name_to_inspect}': {e}")
        logger.error(f"Error fetching chunks for inspection: {e}", exc_info=True)
        return []

# === MAIN STREAMLIT UI ===
st.set_page_config(page_title="Gurudev Satsang Search", layout="wide", initial_sidebar_state="expanded")
st.title("♻️ Gurudev's Words – Satsang Archive Search")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Ingestion Controls")
    uploaded_file = st.file_uploader("Upload .srt transcript", type=["srt"])
    
    st.subheader("📚 Processed Transcripts")
    processed_transcript_list = get_processed_transcripts()
    if processed_transcript_list:
        with st.expander("Show/Hide List", expanded=False):
            for item_name in processed_transcript_list: st.caption(f"• {item_name}")
    else: st.caption("No transcripts processed or unable to fetch.")

    default_transcript_name = Path(uploaded_file.name).stem if uploaded_file else ""
    if 'current_transcript_name_input' not in st.session_state: 
        st.session_state.current_transcript_name_input = default_transcript_name
    if uploaded_file and default_transcript_name != st.session_state.get('last_uploaded_filename_stem_for_input', ''):
        st.session_state.current_transcript_name_input = default_transcript_name
        st.session_state.last_uploaded_filename_stem_for_input = default_transcript_name
    
    user_entered_transcript_name = st.text_input(
        "Transcript Name", 
        value=st.session_state.current_transcript_name_input, 
        key="transcript_name_user_input_field"
    )
    if user_entered_transcript_name != st.session_state.current_transcript_name_input:
        st.session_state.current_transcript_name_input = user_entered_transcript_name
    final_transcript_name_for_processing = st.session_state.current_transcript_name_input

    chunk_size_words_input = st.number_input("Chunk size (target words)", 50, 2000, 400, 50, key="chunk_size_input")
    overlap_words_input = st.number_input("Chunk overlap (target words)", 0, 1000, 75, 25, key="overlap_input")
    if overlap_words_input >= chunk_size_words_input and chunk_size_words_input > 0: 
        st.warning("Overlap < chunk size advised.", icon="⚠️")
    
    process_button_clicked = st.button(
        "Process Transcript (No Entity Tags)", 
        disabled=not uploaded_file or not final_transcript_name_for_processing.strip(), 
        key="process_srt_button"
    )

    st.markdown("---")
    st.header("🏷️ Post-Process Entity Tagging")
    if not CLAUDE_API_KEY:
        st.warning("CLAUDE_API_KEY not set. Tagging disabled.", icon="🔒")
    else:
        if processed_transcript_list:
            selected_transcript_for_tagging = st.selectbox(
                "Select Transcript to Tag Entities:", 
                options=([""] + processed_transcript_list), # Ensure "" is an option
                index=0, 
                key="transcript_select_for_tagging"
            )
            if st.button(
                "Tag Entities for Selected Transcript", 
                disabled=not selected_transcript_for_tagging, 
                key="tag_entities_post_btn"
            ):
                if selected_transcript_for_tagging: # Double check a transcript is selected
                    with st.spinner(f"Initiating tagging for '{selected_transcript_for_tagging}'..."):
                        tag_transcript_entities_post_processing(selected_transcript_for_tagging)
        else: 
            st.caption("Process transcripts to enable tagging.")

    st.markdown("---")
    st.header("🔬 Inspect Transcript Chunks")
    if processed_transcript_list:
        selected_transcript_for_inspection = st.selectbox(
            "Select Transcript to Inspect:",
            options=([""] + processed_transcript_list), # Ensure "" is an option
            index=0,
            key="transcript_select_for_inspection"
        )
        if st.button(
            "Show All Chunks & Payloads", 
            disabled=not selected_transcript_for_inspection, 
            key="inspect_chunks_button"
        ):
            if selected_transcript_for_inspection: # Double check a transcript is selected
                st.session_state.transcript_to_display_chunks = selected_transcript_for_inspection
            else:
                st.session_state.transcript_to_display_chunks = None # Clear if "" selected
    else:
        st.caption("Process some transcripts first to enable inspection.")

# --- Processing Logic (Initial Ingestion) ---
if process_button_clicked:
    # ... (Your existing initial processing logic - ENSURE IT DOES NOT CALL CLAUDE TAGGING)
    st.session_state.qdrant_error_fetching_list = False
    with st.status(f"Processing '{final_transcript_name_for_processing}' (without initial entity tags)...", expanded=True) as status_container:
        st.write("Checking if transcript already exists...")
        logger.info(f"Processing request for transcript: '{final_transcript_name_for_processing}'")
        already_exists_in_qdrant = False
        try:
            scroll_response, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[models.FieldCondition(key="transcript_name", match=models.MatchValue(value=final_transcript_name_for_processing))]),
                limit=1, with_payload=False, with_vectors=False
            )
            already_exists_in_qdrant = bool(scroll_response)
        except Exception as e:
            status_container.update(label=f"Error checking Qdrant: {e}", state="error"); logger.error(f"Qdrant check error: {e}"); st.stop()

        if already_exists_in_qdrant:
            status_container.update(label=f"Transcript '{final_transcript_name_for_processing}' may already exist.", state="warning")
        
        temp_srt_file_path = ""
        try:
            st.write(f"Saving temp SRT for '{uploaded_file.name}'...")
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".srt", delete=False) as temp_f:
                temp_f.write(uploaded_file.getvalue()); temp_srt_file_path = temp_f.name
            
            st.write("Parsing SRT...")
            parsed_subs = parse_srt_file(temp_srt_file_path)
            if not parsed_subs: status_container.update(label="SRT empty/unparsable.", state="error"); st.stop()
            
            st.write(f"Chunking transcript (size: {chunk_size_words_input}, overlap: {overlap_words_input})...")
            transcript_chunks = srt_to_chunks(parsed_subs, chunk_size_words_input, overlap_words_input)
            total_chunks = len(transcript_chunks) # Correct variable name
            if total_chunks == 0: status_container.update(label="No chunks created.", state="error"); st.stop()
            st.write(f"{total_chunks} chunks created.")

            embedding_progress = st.progress(0, text="Embedding & Storing...")
            all_points_for_qdrant = []
            # Corrected loop range in initial processing
            for i in range(0, total_chunks, EMBED_BATCH_SIZE):
                batch_of_chunks = transcript_chunks[i : i + EMBED_BATCH_SIZE] # Corrected slicing
                texts_list = [c['text'] for c in batch_of_chunks]
                batch_vectors = batch_get_embeddings(texts_list)
                points_this_batch = []
                for chunk_idx, (chunk_data, vector_embed) in enumerate(zip(batch_of_chunks, batch_vectors)):
                    actual_idx_overall = i + chunk_idx # Index relative to all chunks
                    if vector_embed is None:
                        embedding_progress.progress((actual_idx_overall + 1) / total_chunks, text=f"Embedding: {actual_idx_overall+1}/{total_chunks} (Skipped)"); continue
                    
                    chunk_payload = {
                        "transcript_name": final_transcript_name_for_processing,
                        "timestamp": chunk_data['timestamp'],
                        "original_text": chunk_data['text']
                        # NO ENTITIES ADDED HERE
                    }
                    points_this_batch.append({'id': str(uuid.uuid4()), 'vector': vector_embed, 'payload': chunk_payload})
                    embedding_progress.progress((actual_idx_overall + 1) / total_chunks, text=f"Embedding: {actual_idx_overall+1}/{total_chunks}")
                if points_this_batch: all_points_for_qdrant.extend(points_this_batch)
            
            if all_points_for_qdrant:
                st.write(f"Upserting {len(all_points_for_qdrant)} points to Qdrant...")
                upsert_to_qdrant(all_points_for_qdrant)
            
            status_container.update(label=f"'{final_transcript_name_for_processing}' processed and stored (without entity tags)!", state="success")
            logger.info(f"Stored transcript: {final_transcript_name_for_processing}")
            get_processed_transcripts.clear() # Refresh list
        except Exception as e:
            status_container.update(label=f"Processing error: {e}", state="error"); logger.error(f"Processing error: {e}", exc_info=True)
        finally:
            if temp_srt_file_path and Path(temp_srt_file_path).exists():
                try: Path(temp_srt_file_path).unlink(); logger.info(f"Deleted temp SRT: {temp_srt_file_path}")
                except OSError as e_del: logger.error(f"Error deleting temp file: {e_del}")


# --- Main Area for Search Results OR Transcript Inspection ---
# This part will now conditionally display either search results or inspected chunks

# Initialize/manage session state for what to display
if 'transcript_to_display_chunks' not in st.session_state:
    st.session_state.transcript_to_display_chunks = None

# If a transcript is selected for inspection, prioritize displaying that
if st.session_state.transcript_to_display_chunks:
    st.markdown("---")
    st.header(f"🔬 Inspecting Chunks for: {st.session_state.transcript_to_display_chunks}")
    
    with st.spinner(f"Fetching all chunks for '{st.session_state.transcript_to_display_chunks}'..."):
        all_chunks_data = get_all_chunks_for_transcript(st.session_state.transcript_to_display_chunks)

    if not all_chunks_data:
        st.info(f"No chunks found in Qdrant for '{st.session_state.transcript_to_display_chunks}' or an error occurred.")
    else:
        st.success(f"Found {len(all_chunks_data)} chunks for '{st.session_state.transcript_to_display_chunks}'.")
        for idx, chunk_info in enumerate(all_chunks_data):
            with st.expander(f"Chunk {idx + 1} (ID: ...{chunk_info['id'][-12:]})", expanded=False):
                st.markdown(f"**Timestamp:** {chunk_info['payload'].get('timestamp', 'N/A')}")
                st.markdown("**Original Text:**")
                st.markdown(f"> {chunk_info['payload'].get('original_text', 'N/A')}")
                
                entities = chunk_info['payload'].get('entities')
                if entities is not None: # Check if entities key exists
                    st.markdown("**Entities (Claude):**")
                    if isinstance(entities, dict) and (entities.get('people') or entities.get('places') or 'self_references' in entities):
                        if entities.get('people'):
                            st.markdown(f"  - **People:** {', '.join(entities['people'])}")
                        if entities.get('places'):
                            st.markdown(f"  - **Places:** {', '.join(entities['places'])}")
                        if 'self_references' in entities:
                            sr_val_inspect = entities.get('self_references')
                            sr_disp_inspect = "N/A"
                            if isinstance(sr_val_inspect, bool): sr_disp_inspect = "Yes" if sr_val_inspect else "No"
                            elif sr_val_inspect is not None: sr_disp_inspect = str(sr_val_inspect)
                            st.markdown(f"  - **Self-ref:** {sr_disp_inspect}")
                    elif isinstance(entities, dict) and not any(entities.values()): # Empty dict like {'people':[], ...}
                        st.caption("  (Claude found no specific entities of interest for this chunk)")
                    else: # Entities key exists but is not a dict or is an unexpected format
                        st.caption(f"  (Entities data present but in unexpected format: {type(entities)})")
                        st.json(entities) # Display raw if not dict
                else:
                    st.caption("  (No Claude entity tags present for this chunk)")

                # Display other payload fields if any (excluding a few common ones already shown)
                other_payload = {k: v for k, v in chunk_info['payload'].items() if k not in ['transcript_name', 'timestamp', 'original_text', 'entities']}
                if other_payload:
                    st.markdown("**Other Payload Data:**")
                    st.json(other_payload)
    
    # Add a button to clear the inspection view and go back to search
    if st.button("Clear Inspection (Back to Search)", key="clear_inspection_view_btn"):
        st.session_state.transcript_to_display_chunks = None
        st.rerun() # Rerun to update the main display area

# Only show search UI if not inspecting chunks
elif not st.session_state.transcript_to_display_chunks:
    st.markdown("---")
    st.header("🔍 Semantic Search & Analysis")
    search_query_input = st.text_input("Enter your search query:", key="main_search_query_input")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1: use_llm_reranking = st.checkbox("Enable LLM Reranking", value=True, key="rerank_toggle_checkbox", help=f"Uses {RERANK_MODEL}")
    with col_opt2: do_pinpoint_answer_extraction = st.checkbox("Pinpoint Answer Snippet", value=True, key="pinpoint_answer_checkbox", help=f"Uses {ANSWER_EXTRACTION_MODEL}")
    with col_opt3: initial_search_results_limit = st.slider("Initial results:", 3, 30, 5, key="search_results_limit_slider")
    custom_reranking_instructions_input = ""
    if use_llm_reranking: custom_reranking_instructions_input = st.text_area("Custom Reranking Instructions (Optional):", placeholder="e.g., 'Prioritize practical advice.'", key="custom_rerank_instructions_input_area", height=100)

    if search_query_input:
        with st.spinner("Searching and analyzing..."):
            # 1. Get Query Embedding
            try: query_vector = client.embeddings.create(model=EMBEDDING_MODEL, input=[search_query_input]).data[0].embedding
            except Exception as e: st.error(f"Query embed fail: {e}"); logger.error(f"Query embed error: {e}"); st.stop()
            # 2. Search Qdrant
            try:
                qdrant_hits = qdrant_client.search(COLLECTION_NAME, query_vector, limit=initial_search_results_limit, with_payload=True)
            except Exception as e: st.error(f"Qdrant search fail: {e}"); logger.error(f"Qdrant search error: {e}"); st.stop()
            retrieved_payloads = [h.payload for h in qdrant_hits if h.payload]
            if not retrieved_payloads: st.info("No results found."); st.stop()

            # 3. Reranking
            if use_llm_reranking and len(retrieved_payloads) > 1:
                with st.spinner(f"Reranking with {RERANK_MODEL}..."):
                    excerpts_for_llm_rerank = [f"[{i+1}] {rp.get('original_text', 'N/A')}" for i, rp in enumerate(retrieved_payloads)]
                    instr_prefix = "Rerank by relevance to query."
                    if custom_reranking_instructions_input: instr_prefix += f" Criteria: {custom_reranking_instructions_input}."
                    prompt = (f"{instr_prefix}\nQuery: \"{search_query_input}\"\nExcerpts:\n{chr(10).join(excerpts_for_llm_rerank)}\nReturn only comma-separated original indices reranked.")
                    try:
                        resp = client.chat.completions.create(model=RERANK_MODEL, messages=[{"role":"system", "content":"You rerank results."}, {"role":"user", "content":prompt}], max_tokens=200, temperature=0.0)
                        indices_str = resp.choices[0].message.content.strip()
                        parsed_indices = []; seen = set()
                        for s_idx in re.findall(r'\d+', indices_str):
                            val = int(s_idx) - 1
                            if 0 <= val < len(retrieved_payloads) and val not in seen: parsed_indices.append(val); seen.add(val)
                        if parsed_indices:
                            final_order = parsed_indices + [i for i in range(len(retrieved_payloads)) if i not in seen]
                            retrieved_payloads = [retrieved_payloads[i] for i in final_order]
                            st.caption(f"ℹ️ Results reranked by {RERANK_MODEL}.")
                        else: st.warning("Reranking failed/no change. Original order.", icon="⚠️")
                    except Exception as e_rr: st.warning(f"Reranking error: {e_rr}. Original order.", icon="⚠️"); logger.error(f"Reranking error: {e_rr}")
            
            # 4. Display Results
            st.subheader(f"Search Results for: \"{search_query_input}\"")
            for i, payload_data in enumerate(retrieved_payloads, 1):
                if not isinstance(payload_data, dict): continue 
                full_text = payload_data.get('original_text', 'Error: Text missing')
                exp_title = f"Result {i} @ {payload_data.get('timestamp','N/A')} | Transcript: {payload_data.get('transcript_name','N/A')}"
                with st.expander(exp_title, expanded=(i==1)):
                    if do_pinpoint_answer_extraction and full_text != 'Error: Text missing':
                        with st.spinner(f"Pinpointing answer for result {i}..."):
                            answer_span = extract_answer_span(search_query_input, full_text, ANSWER_EXTRACTION_MODEL)
                        if answer_span and answer_span.strip() and answer_span.strip().lower() != full_text.strip().lower() and len(answer_span.strip()) < len(full_text.strip()):
                            st.markdown(f"🎯 **Pinpointed Answer Snippet:**\n> *{answer_span.strip()}*"); st.markdown("--- \n**Full Context Chunk:**")
                        elif answer_span and answer_span.strip().lower() == full_text.strip().lower(): st.caption(f"(Full chunk identified as most direct answer)")
                    st.markdown(full_text)
                    entities_payload = payload_data.get('entities') 
                    # Debugging: st.write(f"DEBUG search - entities_payload for result {i}: {entities_payload}")
                    if entities_payload and isinstance(entities_payload, dict):
                        entity_parts_disp = []
                        if entities_payload.get('people'): entity_parts_disp.append(f"**People:** {', '.join(entities_payload['people'])}")
                        if entities_payload.get('places'): entity_parts_disp.append(f"**Places:** {', '.join(entities_payload['places'])}")
                        if 'self_references' in entities_payload: 
                            sr_val = entities_payload.get('self_references')
                            sr_disp = "N/A"
                            if isinstance(sr_val, bool): sr_disp = "Yes" if sr_val else "No"
                            elif sr_val is not None: sr_disp = str(sr_val)
                            entity_parts_disp.append(f"**Self-ref:** {sr_disp}")
                        if entity_parts_disp: st.markdown("---"); st.markdown(" ".join(entity_parts_disp), unsafe_allow_html=True)
                    # else:
                        # st.caption("(No entity tags found for this chunk in search result)") # Optional debug
app.py
Displaying app.py.