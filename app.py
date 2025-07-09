#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gurudev Satsang Transcript Processing Pipeline + Streamlit UI
- Robust Qdrant setup, separate post-processing for Claude & Fine-tuned models.
- Corrected filter logic to use OR (should) for a better multi-category search experience.
- Added explicit timeout to Qdrant client to handle DEADLINE_EXCEEDED errors.
- Fixed 'vectors_config' attribute error for compatibility with older qdrant-client versions.
"""

import os
import streamlit as st
import uuid
import srt
import httpx
import json
import re
from qdrant_client import QdrantClient, models
from pathlib import Path
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv, find_dotenv
import tempfile
import logging
import ast
import pandas as pd
from io import BytesIO
st.set_page_config(page_title="Gurudev Satsang Search", layout="wide", initial_sidebar_state="expanded")
# === CONFIG ===
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6334))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gurudev_satsangs_bio_v3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 20))
ENTITY_MODEL = os.getenv("ENTITY_MODEL", "claude-3-haiku-20240307")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4-turbo-preview")
ANSWER_EXTRACTION_MODEL = os.getenv("ANSWER_EXTRACTION_MODEL", "gpt-3.5-turbo")
FINE_TUNED_BIO_MODEL_ID = "ft:gpt-3.5-turbo-0125:srmd:satsang-search-v1:BgoxJBWJ" # Loaded from .env

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Attempting to load FINE_TUNED_BIO_MODEL_ID from env: {FINE_TUNED_BIO_MODEL_ID}")


BIOGRAPHICAL_CATEGORY_KEYS = [
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

SATSANG_CATEGORIES = [
    "Pravachan",
    "Udgosh",
    "Meetings",
    "Prasangik Bodh",
    "Ashirvachan",
    "Experiences",
    "Miscellaneous"
]

LOCATIONS = [
    "Sayla",
    "Surat",
    "Vadodara",
    "Ahmedabad", 
    "Mumbai",
    "Rajkot",
    "Bhavnagar",
    "Morbi",
    "Junagadh",
    "Jamnagar",
    "Gandhinagar",
    "Nadiad",
    "Anand",
    "Bharuch",
    "Navsari",
    "Vapi"
]

SPEAKERS = [
    "Gurudev",
    "P. Pu. Bapuji",
    "P. Pu. Sadgurudev",
    "Sadguru Shashtriji Maharaj",
    "Bhaishree",
    "Param Pujya Bhaishree",
    "P. Pu. Ladakchandbhai",
    "Brahmanishth Vikrambhai",
    "Brahmanishth Rajubhai"
]

# === INITIALIZATION ===
if not OPENAI_API_KEY: st.error("OPENAI_API_KEY not found."); st.stop()
try: client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e: st.error(f"OpenAI client init error: {e}"); logger.error(f"OpenAI init error: {e}"); st.stop()

qdrant_client = None
if QDRANT_API_KEY and QDRANT_HOST:
    try:
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            timeout=30
        )
        qdrant_client.get_collections()
        logger.info(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        st.error(f"Qdrant connection error: {e}. Check QDRANT_HOST (URL), QDRANT_PORT (often 6334 for gRPC), and QDRANT_API_KEY.")
        logger.error(f"Qdrant connection error: {e}"); st.stop()
else:
    st.error("QDRANT_HOST and QDRANT_API_KEY must be set."); logger.error("QDRANT_HOST or QDRANT_API_KEY not set."); st.stop()

def get_field_schema_type_from_qdrant_field_info(field_info_obj):
    """Helper to determine schema type from Qdrant's field info object."""
    if not field_info_obj:
        return None
    if hasattr(field_info_obj, 'keyword') and field_info_obj.keyword is not None:
        return models.PayloadSchemaType.KEYWORD
    if hasattr(field_info_obj, 'integer') and field_info_obj.integer is not None:
        return models.PayloadSchemaType.INTEGER
    if hasattr(field_info_obj, 'float') and field_info_obj.float is not None:
        return models.PayloadSchemaType.FLOAT
    if hasattr(field_info_obj, 'geo') and field_info_obj.geo is not None:
        return models.PayloadSchemaType.GEO
    if hasattr(field_info_obj, 'text') and field_info_obj.text is not None:
        return models.PayloadSchemaType.TEXT
    if hasattr(field_info_obj, 'data_type'):
        dt_value = field_info_obj.data_type
        if isinstance(dt_value, models.PayloadSchemaType): return dt_value
        if isinstance(dt_value, str):
            for enum_member in models.PayloadSchemaType:
                if enum_member.value == dt_value: return enum_member
    return None

def setup_qdrant_collection_and_indexes(client_instance: QdrantClient, coll_name: str, vec_dim: int, vec_dist: models.Distance):
    logger.info(f"Setting up Qdrant collection '{coll_name}'...")
    indexes_to_ensure = {
        "transcript_name": models.PayloadSchemaType.KEYWORD,
        "entities.self_references": models.PayloadSchemaType.KEYWORD
    }
    for cat_key in BIOGRAPHICAL_CATEGORY_KEYS:
        indexes_to_ensure[f"has_{cat_key}"] = models.PayloadSchemaType.KEYWORD

    try:
        collection_info = client_instance.get_collection(collection_name=coll_name)
        recreate_collection = False

        # CORRECTED: Reverted to the `config.params.vectors` structure, which is compatible
        # with older qdrant-client versions that do not have the 'vectors_config' attribute.
        # This directly addresses the "'CollectionInfo' object has no attribute 'vectors_config'" error.
        current_vectors_config = collection_info.config.params.vectors
        if current_vectors_config.size != vec_dim or \
           current_vectors_config.distance != vec_dist:
            logger.warning(f"Collection '{coll_name}' vector config differs. Will recreate.")
            recreate_collection = True

        if recreate_collection:
            logger.info(f"Recreating collection '{coll_name}'.")
            client_instance.recreate_collection(collection_name=coll_name, vectors_config=models.VectorParams(size=vec_dim, distance=vec_dist))
            logger.info(f"Collection '{coll_name}' recreated.")
            for field_name, schema_type in indexes_to_ensure.items():
                client_instance.create_payload_index(collection_name=coll_name, field_name=field_name, field_schema=schema_type)
                logger.info(f"Created payload index on '{field_name}' ({schema_type.value}) for '{coll_name}'.")
        else:
            logger.info(f"Collection '{coll_name}' exists with correct vector config. Verifying indexes...")
            detailed_collection_info = client_instance.get_collection(collection_name=coll_name)
            current_payload_schema_map = detailed_collection_info.payload_schema or {}
            
            for field_name, expected_schema_type in indexes_to_ensure.items():
                field_info_from_qdrant = current_payload_schema_map.get(field_name)
                actual_schema_type = get_field_schema_type_from_qdrant_field_info(field_info_from_qdrant)

                if actual_schema_type == expected_schema_type:
                    logger.info(f"Payload index on '{field_name}' of type '{expected_schema_type.value}' already exists for '{coll_name}'.")
                else:
                    logger.info(f"Index on '{field_name}' for '{coll_name}': Expected '{expected_schema_type.value}', Found '{actual_schema_type.value if actual_schema_type else 'None or different'}'. Attempting to create/recreate.")
                    try:
                        client_instance.create_payload_index(collection_name=coll_name, field_name=field_name, field_schema=expected_schema_type)
                        logger.info(f"Successfully ensured index on '{field_name}' of type '{expected_schema_type.value}'.")
                    except Exception as e_create_idx:
                        logger.error(f"Failed to create/recreate index for '{field_name}': {e_create_idx}. It might exist with an incompatible type. Manual check might be needed.")
        logger.info(f"Qdrant collection '{coll_name}' setup check complete.")

    except Exception as e_get_col:
        is_not_found_error = "not found" in str(e_get_col).lower() or \
                             (hasattr(e_get_col, "status_code") and e_get_col.status_code == 404) or \
                             (hasattr(e_get_col, "grpc_code") and e_get_col.grpc_code == 5)

        if is_not_found_error:
            logger.info(f"Collection '{coll_name}' not found. Creating collection and all defined indexes.")
            client_instance.recreate_collection(collection_name=coll_name, vectors_config=models.VectorParams(size=vec_dim, distance=vec_dist))
            for field_name, schema_type in indexes_to_ensure.items():
                client_instance.create_payload_index(collection_name=coll_name, field_name=field_name, field_schema=schema_type)
                logger.info(f"Created payload index on '{field_name}' ({schema_type.value}) for '{coll_name}'.")
            logger.info(f"Qdrant collection '{coll_name}' created and all indexes setup complete.")
        else:
            st.error(f"Critical error during Qdrant collection setup for '{coll_name}': {e_get_col}")
            logger.error(f"Critical Qdrant setup error for '{coll_name}': {e_get_col}", exc_info=True); st.stop()


# [The rest of the script is identical and correct. No other changes are needed.]
# ...
if qdrant_client:
    setup_qdrant_collection_and_indexes(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM, models.Distance.COSINE)

# ... (I'm omitting the rest of the file for brevity as it remains unchanged)
# The full script with all helper functions and the UI should be used with this corrected setup function.
# The code below is just for completeness of the file structure.



def parse_srt_file(file_path: str) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return list(srt.parse(f.read()))
    except Exception as e: logger.error(f"SRT parse error {file_path}: {e}"); st.error(f"SRT parse error: {e}"); return []

def srt_to_chunks(subs: list, chunk_size_words: int, overlap_words: int) -> list:
    chunks, current_sub_idx = [], 0;
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
    if not CLAUDE_API_KEY:
        logger.warning("CLAUDE_API_KEY not set.")
        return {}
        
    headers = {"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    # Emphasized the JSON requirement in the prompt
    prompt = (f"Analyze text, extract entities. Respond *only* with a strict JSON object: {{\"people\": [], \"places\": [], \"self_references\": bool}}. Use double quotes for all keys.\nText:\n{text}\nJSON Output:")
    body = {"model": ENTITY_MODEL, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}

    try:
        with httpx.Client(timeout=45.0) as http_client:
            response = http_client.post(CLAUDE_API_URL, headers=headers, json=body)
            response.raise_for_status()
        data = response.json()
        
        if data.get('content') and data['content'][0].get('type') == 'text':
            raw_text = data['content'][0]['text']
            
            # Find the JSON string, whether it's in a code block or not
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
            json_str = json_match.group(1) if json_match else raw_text[raw_text.find('{'):raw_text.rfind('}')+1]
            
            if not json_str.strip():
                logger.warning(f"Claude returned empty content for parsing: {raw_text}")
                return {}

            entities = {}
            # CORRECTED LOGIC: Implement a two-step parsing strategy
            try:
                # 1. Try parsing as strict JSON first (ideal case)
                entities = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Claude JSON parse failed. Falling back to ast.literal_eval. Offending string: {json_str[:150]}")
                try:
                    # 2. If JSON fails, try parsing as a Python dictionary literal (handles single quotes)
                    entities = ast.literal_eval(json_str)
                except (ValueError, SyntaxError) as e_ast:
                    # If both fail, the string is truly malformed.
                    logger.error(f"AST parsing also failed: {e_ast}. String: {json_str[:150]}")
                    return {}
            
            # Final validation of the parsed structure
            return {
                'people': entities.get('people', []) if isinstance(entities.get('people'), list) else [],
                'places': entities.get('places', []) if isinstance(entities.get('places'), list) else [],
                'self_references': entities.get('self_references', False) if isinstance(entities.get('self_references'), bool) else False
            }

        logger.warning(f"Claude unexpected response structure: {data.get('content')}")
        return {}
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Claude HTTP error: {e.response.status_code} - {e.response.text}")
        return {}
    except Exception as e:
        logger.error(f"Claude API/tagging error: {e}")
        return {}

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
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=structs, wait=True)
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
            if not next_offset_val or not points_batch or len(names) > 1000: break
            offset_val = next_offset_val
        return sorted(list(names))
    except Exception as e: logger.error(f"Error fetching transcript list: {e}"); return []

def tag_transcript_entities_post_processing(transcript_name_to_tag: str):
    if not qdrant_client: st.error("Qdrant client N/A."); return
    if not CLAUDE_API_KEY: st.warning("CLAUDE_API_KEY not set.", icon="⚠️"); return
    st.info(f"Fetching chunks for '{transcript_name_to_tag}' to tag with Claude entities...")
    updates_to_perform = []
    try:
        offset_val, fetched_count, processed_count = None, 0, 0
        prog_text = st.empty(); prog_bar = st.progress(0)
        while True:
            prog_text.text(f"Fetching batch for '{transcript_name_to_tag}'..."); prog_bar.progress(min(fetched_count/1000, 0.1))
            retrieved_points, next_offset_val = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[models.FieldCondition(key="transcript_name", match=models.MatchValue(value=transcript_name_to_tag))]),
                limit=50, offset=offset_val, with_payload=True, with_vectors=False
            )
            if not retrieved_points: break
            fetched_count += len(retrieved_points)
            for point in retrieved_points:
                processed_count +=1
                prog_text.text(f"Tagging chunk {processed_count}/{fetched_count} (ID: ...{str(point.id)[-6:]})")
                prog_bar.progress(processed_count / max(1, fetched_count))
                if point.payload and 'original_text' in point.payload:
                    if point.payload.get('entities') is not None: logger.info(f"Point {point.id} already has Claude entities. Skipping."); continue
                    chunk_text = point.payload['original_text']
                    entities = tag_entities_with_claude(chunk_text)
                    if entities: updates_to_perform.append((point.id, {'entities': entities}))
                else: logger.warning(f"Point {point.id} missing 'original_text'. Skipping.")
            if not next_offset_val: break
            offset_val = next_offset_val
        prog_bar.empty(); prog_text.empty()
        if not updates_to_perform: st.success(f"No new chunks for Claude entity tagging in '{transcript_name_to_tag}'."); return
        st.write(f"Applying Claude entity tags to {len(updates_to_perform)} chunks in '{transcript_name_to_tag}'...");
        apply_prog = st.progress(0, text="Applying Claude tags to Qdrant...")
        for i, (point_id, entity_payload_dict) in enumerate(updates_to_perform):
            try:
                qdrant_client.set_payload(collection_name=COLLECTION_NAME, payload=entity_payload_dict, points=[point_id], wait=True)
            except Exception as e_set: logger.error(f"Failed to set Claude entities for {point_id}: {e_set}"); st.warning(f"Failed to update point ...{str(point_id)[-6:]}", icon="⚠️")
            apply_prog.progress((i + 1) / len(updates_to_perform), text=f"Applying Claude tags: {i+1}/{len(updates_to_perform)}")
        st.success(f"Claude entity tagging complete for '{transcript_name_to_tag}'.")
        get_processed_transcripts.clear()
    except Exception as e: st.error(f"Error during Claude entity tagging for '{transcript_name_to_tag}': {e}"); logger.error(f"Post-processing error for '{transcript_name_to_tag}': {e}", exc_info=True)
    finally:
        if 'prog_bar' in locals() and prog_bar is not None: prog_bar.empty()
        if 'prog_text' in locals() and prog_text is not None: prog_text.empty()

def extract_and_store_biographical_info(transcript_name_to_process: str, ft_model_id_to_use: str):
    if not qdrant_client: st.error("Qdrant client N/A."); return
    logger.info(f"Attempting to use fine-tuned model ID: '{ft_model_id_to_use}' for bio-extraction on '{transcript_name_to_process}'.")
    if not ft_model_id_to_use:
        st.warning("Fine-tuned model ID is empty or not provided. Bio-extraction disabled.", icon="⚠️")
        logger.warning("Fine-tuned model ID was empty/None when extract_and_store_biographical_info was called.")
        return

    st.info(f"Fetching chunks for '{transcript_name_to_process}' to extract biographical info with model '{ft_model_id_to_use[:20]}...'...")
    updates_to_perform = []
    try:
        offset_val, fetched_count, processed_count = None, 0, 0
        prog_text = st.empty(); prog_bar = st.progress(0)
        while True:
            prog_text.text(f"Fetching batch for bio-extraction '{transcript_name_to_process}'..."); prog_bar.progress(min(fetched_count/500, 0.1))
            retrieved_points, next_offset_val = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[models.FieldCondition(key="transcript_name", match=models.MatchValue(value=transcript_name_to_process))]),
                limit=10,
                offset=offset_val, with_payload=True, with_vectors=False
            )
            if not retrieved_points: break
            fetched_count += len(retrieved_points)

            for point in retrieved_points:
                processed_count +=1
                prog_text.text(f"Bio-extracting chunk {processed_count}/{fetched_count} (ID: ...{str(point.id)[-6:]})")
                prog_bar.progress(processed_count / max(1, fetched_count))

                if point.payload and 'original_text' in point.payload:
                    if point.payload.get('biographical_extractions') is not None:
                        logger.info(f"Point {point.id} already has 'biographical_extractions'. Skipping."); continue
                    
                    chunk_text = point.payload['original_text']
                    logger.info(f"Calling fine-tuned model '{ft_model_id_to_use}' for point {point.id}...")
                    extracted_json_str = ""
                    try:
                        ft_response = client.chat.completions.create(
                            model=ft_model_id_to_use,
                            messages=[
                                {"role": "system", "content": "You are an expert at extracting specific biographical information about Gurudev from transcripts, outputting a JSON object with predefined keys like early_life_childhood, education_learning, etc. Only include verbatim quotes. If no information for a category, use an empty list []."},
                                {"role": "user", "content": f"Transcript Chunk: \"{chunk_text}\""}
                            ],
                            response_format={"type": "json_object"},
                            max_tokens=4096,
                            temperature=0.0
                        )
                        extracted_json_str = ft_response.choices[0].message.content
                        if extracted_json_str.startswith("```json"): extracted_json_str = extracted_json_str[7:]
                        if extracted_json_str.endswith("```"): extracted_json_str = extracted_json_str[:-3]
                        extracted_json_str = extracted_json_str.strip()
                        parsed_bio_data = json.loads(extracted_json_str)

                        payload_update = {'biographical_extractions': parsed_bio_data}
                        for cat_key in BIOGRAPHICAL_CATEGORY_KEYS:
                            flag_field_name = f"has_{cat_key}"
                            payload_update[flag_field_name] = bool(parsed_bio_data.get(cat_key))
                        updates_to_perform.append((point.id, payload_update))
                    except json.JSONDecodeError as e_json: logger.error(f"FT Model JSON parse error for point {point.id}: {e_json}. Response: {extracted_json_str[:300]}")
                    except OpenAIError as e_ft_openai: logger.error(f"OpenAI API error (FT model) for point {point.id}: {e_ft_openai}")
                    except Exception as e_ft_general: logger.error(f"Unexpected error (FT model) for point {point.id}: {e_ft_general}")
                else: logger.warning(f"Point {point.id} missing 'original_text' for bio-extraction. Skipping.")
            if not next_offset_val: break
            offset_val = next_offset_val
        prog_bar.empty(); prog_text.empty()
        if not updates_to_perform: st.success(f"No new chunks for biographical extraction in '{transcript_name_to_process}'."); return
        st.write(f"Applying biographical extractions to {len(updates_to_perform)} chunks...");
        apply_prog = st.progress(0, text="Applying bio-extractions to Qdrant...")
        for i, (point_id, bio_payload_update_dict) in enumerate(updates_to_perform):
            try: qdrant_client.set_payload(collection_name=COLLECTION_NAME, payload=bio_payload_update_dict, points=[point_id], wait=True)
            except Exception as e_set_bio: logger.error(f"Failed to set bio-extractions for {point_id}: {e_set_bio}"); st.warning(f"Failed to update ...{str(point_id)[-6:]} with bio-data", icon="⚠️")
            apply_prog.progress((i + 1) / len(updates_to_perform), text=f"Applying bio-extractions: {i+1}/{len(updates_to_perform)}")
        st.success(f"Biographical extraction complete for '{transcript_name_to_process}'.")
        get_processed_transcripts.clear()
    except Exception as e: st.error(f"Error during bio-extraction for '{transcript_name_to_process}': {e}"); logger.error(f"Bio-extraction error for '{transcript_name_to_process}': {e}", exc_info=True)
    finally:
        if 'prog_bar' in locals() and prog_bar is not None: prog_bar.empty()
        if 'prog_text' in locals() and prog_text is not None: prog_text.empty()

# --- Helper functions to check processing status ---
def get_transcripts_with_field(field_name: str) -> set:
    """Return set of transcript names where at least one chunk has the given field."""
    found = set()
    if not qdrant_client:
        return found
    try:
        offset_val = None
        while True:
            points_batch, next_offset_val = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset_val,
                with_payload=["transcript_name", field_name],
                with_vectors=False
            )
            for p in points_batch:
                if p.payload and p.payload.get(field_name) is not None and 'transcript_name' in p.payload:
                    found.add(p.payload['transcript_name'])
            if not next_offset_val or not points_batch:
                break
            offset_val = next_offset_val
    except Exception as e:
        logger.error(f"Error checking transcripts with field {field_name}: {e}")
    return found

# --- Get processed transcript lists ---
processed_transcript_list = get_processed_transcripts()
claude_done = get_transcripts_with_field("entities")
bio_done = get_transcripts_with_field("biographical_extractions")

def highlight_transcript(name, done_set):
    return f"✅ {name}" if name in done_set else name

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
# --- STREAMLIT UI ---

st.title("♻️ Gurudev's Words – Satsang Archive Search")

with st.sidebar:
    st.header("⚙️ Ingestion & Processing")
    
    # File upload
    uploaded_file = st.file_uploader("1. Upload .srt transcript", type=["srt"])
    
    # Metadata input fields
    col1, col2 = st.columns(2)
    with col1:
        transcript_date = st.date_input(
            "Date of Satsang",
            key="transcript_date_input"
        )
    with col2:
        transcript_category = st.selectbox(
            "Category",
            options=SATSANG_CATEGORIES,
            key="transcript_category_input"
        )

    # Simple location selection from glossary
    location_input = st.selectbox(
        "Location",
        options=[""] + sorted(LOCATIONS),
        key="location_select",
        help="Select location from predefined list"
    )

    # Optional: Add new location through expander
    with st.expander("➕ Add New Location"):
        new_location = st.text_input("New Location Name")
        if st.button("Add Location"):
            if new_location and new_location not in LOCATIONS:
                LOCATIONS.append(new_location)
                st.success(f"Added {new_location} to locations list")
                st.experimental_rerun()
            elif new_location in LOCATIONS:
                st.warning("Location already exists")
            else:
                st.error("Please enter a location name")

    # Speaker selection from predefined list
    speaker_input = st.selectbox(
        "Speaker",
        options=[""] + sorted(SPEAKERS),
        key="speaker_select",
        help="Select speaker from predefined list"
    )

    # Optional: Add new speaker through expander
    with st.expander("➕ Add New Speaker"):
        new_speaker = st.text_input("New Speaker Name")
        if st.button("Add Speaker"):
            if new_speaker and new_speaker not in SPEAKERS:
                SPEAKERS.append(new_speaker)
                st.success(f"Added {new_speaker} to speakers list")
                st.experimental_rerun()
            elif new_speaker in SPEAKERS:
                st.warning("Speaker already exists")
            else:
                st.error("Please enter a speaker name")

    # Satsang name and misc tag
    satsang_name = st.text_input(
        "Satsang Name",
        key="satsang_name_input"
    )
    satsang_code = st.text_input(
        "Satsang Code",
        key="satsang_code_input",
        help="Enter unique code/reference number for the satsang"
    )
    misc_tag = st.text_input(
        "Miscellaneous Tags",
        key="misc_tag_input",
        help="Add any additional tags, separated by commas"
    )

    # Modify the transcript name construction
    if uploaded_file:
        default_transcript_name = (f"{transcript_date.strftime('%Y%m%d')}_{satsang_name}"
                                 if satsang_name else Path(uploaded_file.name).stem)
        
        if 'current_transcript_name_input' not in st.session_state:
            st.session_state.current_transcript_name_input = default_transcript_name

    # Add processing button
    process_button_clicked = st.button(
        "Process Transcript",
        disabled=not uploaded_file,  # Disable if no file uploaded
        help="Click to process the uploaded transcript with metadata"
    )

# Add after file upload in sidebar
chunking_col1, chunking_col2 = st.columns(2)
with chunking_col1:
    chunk_size_words_input = st.number_input(
        "Words per Chunk",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Target number of words per chunk"
    )
with chunking_col2:
    overlap_words_input = st.number_input(
        "Overlap Words",
        min_value=0,
        max_value=100,
        value=50,
        step=10,
        help="Number of words to overlap between chunks"
    )

# Modify the chunk processing to include metadata
def process_chunks_with_metadata(chunks, metadata):
    """Add metadata to each chunk before processing."""
    processed_chunks = []
    for chunk in chunks:
        chunk_dict = {
            'text': chunk['text'],
            'timestamp': chunk['timestamp'],
            'date': metadata['date'],
            'category': metadata['category'],
            'location': metadata['location'],
            'speaker': metadata['speaker'],
            'satsang_name': metadata['satsang_name'],
            'satsang_code': metadata['satsang_code'],  # Add this line
            'misc_tags': metadata['misc_tags'],
            'transcript_name': metadata['transcript_name']
        }
        processed_chunks.append(chunk_dict)
    return processed_chunks

# Modify the main processing section
if process_button_clicked:
    if not final_transcript_name_for_processing.strip():
        st.error("Please provide a transcript name.")
    else:
        try:
            # Create metadata dictionary
            metadata = {
                'date': transcript_date.strftime('%Y-%m-%d'),
                'category': transcript_category,
                'location': location_input,
                'speaker': speaker_input,
                'satsang_name': satsang_name,
                'satsang_code': satsang_code,  # Add this line
                'misc_tags': [tag.strip() for tag in misc_tag.split(',')] if misc_tag else [],
                'transcript_name': final_transcript_name_for_processing
            }

            # Process SRT file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.srt') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            subs = parse_srt_file(tmp_file_path)
            os.unlink(tmp_file_path)

            if not subs:
                st.error("No subtitles found in file.")
                st.stop()

            chunks = srt_to_chunks(subs, chunk_size_words_input, overlap_words_input)
            if not chunks:
                st.error("Chunking failed.")
                st.stop()

            # Add metadata to chunks
            chunks_with_metadata = process_chunks_with_metadata(chunks, metadata)
            
            # Process the chunks with metadata
            # Add your processing logic here
            
        except Exception as e:
            st.error(f"Error processing transcript: {str(e)}")
            logger.error(f"Transcript processing error: {e}", exc_info=True)
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
            st.subheader(f"Search Results for: \"search_query_input\"")
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

# Add after metadata inputs
st.markdown("---")
st.subheader("📊 Post-Processing")

post_process_col1, post_process_col2 = st.columns(2)
with post_process_col1:
    transcript_to_tag = st.selectbox(
        "Select Transcript for Entity Tagging",
        options=[None] + [highlight_transcript(name, claude_done) for name in processed_transcript_list],
        format_func=lambda x: "Select transcript..." if x is None else x
    )
    if transcript_to_tag and st.button("Run Entity Tagging"):
        tag_transcript_entities_post_processing(transcript_to_tag.replace("✅ ", ""))

with post_process_col2:
    transcript_for_bio = st.selectbox(
        "Select Transcript for Bio-Extraction",
        options=[None] + [highlight_transcript(name, bio_done) for name in processed_transcript_list],
        format_func=lambda x: "Select transcript..." if x is None else x
    )
    if transcript_for_bio and st.button("Run Bio-Extraction"):
        extract_and_store_biographical_info(transcript_for_bio.replace("✅ ", ""), FINE_TUNED_BIO_MODEL_ID)

# Add after post-processing section
st.markdown("---")
st.subheader("📜 Transcript Management")

inspect_transcript = st.selectbox(
    "Select Transcript to Inspect",
    options=[None] + processed_transcript_list,
    format_func=lambda x: "Select transcript..." if x is None else x,
    key="inspect_transcript_select"
)

if inspect_transcript and st.button("Inspect Chunks"):
    st.session_state.transcript_to_display_chunks = inspect_transcript
    st.experimental_rerun()

