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
import ast

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

# === MAIN STREAMLIT UI ===
st.set_page_config(page_title="Gurudev Satsang Search", layout="wide", initial_sidebar_state="expanded")
st.title("♻️ Gurudev's Words – Satsang Archive Search")

with st.sidebar:
    st.header("⚙️ Ingestion & Processing")
    uploaded_file = st.file_uploader("1. Upload .srt transcript", type=["srt"])
    st.subheader("📚 Processed Transcripts")
    processed_transcript_list = get_processed_transcripts()
    if processed_transcript_list:
        with st.expander("Show/Hide List", expanded=False):
            for item_name in processed_transcript_list: st.caption(f"• {item_name}")
    else: st.caption("No transcripts processed or unable to fetch.")

    default_transcript_name = Path(uploaded_file.name).stem if uploaded_file else ""
    if 'current_transcript_name_input' not in st.session_state: st.session_state.current_transcript_name_input = default_transcript_name
    if uploaded_file and default_transcript_name != st.session_state.get('last_uploaded_filename_stem_for_input', ''):
        st.session_state.current_transcript_name_input = default_transcript_name
        st.session_state.last_uploaded_filename_stem_for_input = default_transcript_name
    user_entered_transcript_name = st.text_input("Transcript Name (for processing):", value=st.session_state.current_transcript_name_input, key="transcript_name_user_input_field")
    if user_entered_transcript_name != st.session_state.current_transcript_name_input: st.session_state.current_transcript_name_input = user_entered_transcript_name
    final_transcript_name_for_processing = st.session_state.current_transcript_name_input

    chunk_size_words_input = st.number_input("Chunk size (words)", 50, 2000, 400, 50, key="chunk_size_input")
    overlap_words_input = st.number_input("Chunk overlap (words)", 0, 1000, 75, 25, key="overlap_input")
    if overlap_words_input >= chunk_size_words_input and chunk_size_words_input > 0: st.warning("Overlap < chunk size advised.", icon="⚠️")
    
    st.markdown("---")
    st.markdown("**Initial Processing (Embed & Store):**")
    process_button_clicked = st.button("2. Process Transcript (Embeds Chunks)", disabled=not uploaded_file or not final_transcript_name_for_processing.strip(), key="process_srt_button")

    st.markdown("---")
    st.subheader("🏷️ Post-Processing Steps")
    st.markdown("**Claude Entity Tagging:**")
    if not CLAUDE_API_KEY: st.warning("CLAUDE_API_KEY not set.", icon="🔒")
    else:
        if processed_transcript_list:
            selected_transcript_for_claude_tagging = st.selectbox("Select Transcript for Claude Entities:", options=[""] + processed_transcript_list, index=0, key="transcript_select_for_claude_tagging")
            if st.button("3a. Tag with Claude Entities", disabled=not selected_transcript_for_claude_tagging, key="claude_tag_button"):
                with st.spinner(f"Initiating Claude entity tagging for '{selected_transcript_for_claude_tagging}'..."):
                    tag_transcript_entities_post_processing(selected_transcript_for_claude_tagging)
        else: st.caption("Process transcripts first.")

    st.markdown("**Fine-Tuned Biographical Extraction:**")
    if not FINE_TUNED_BIO_MODEL_ID:
        st.caption("FINE_TUNED_BIO_MODEL_ID not set in env. Bio-extraction disabled.")
    else:
        if processed_transcript_list:
            selected_transcript_for_bio_extraction = st.selectbox("Select Transcript for Bio-Extraction:", options=[""] + processed_transcript_list, index=0, key="transcript_select_for_bio_extraction")
            if st.button("3b. Extract Biographical Details (Fine-tuned Model)", disabled=not selected_transcript_for_bio_extraction, key="bio_extract_button"):
                with st.spinner(f"Initiating fine-tuned bio-extraction for '{selected_transcript_for_bio_extraction}'..."):
                    extract_and_store_biographical_info(selected_transcript_for_bio_extraction, FINE_TUNED_BIO_MODEL_ID)
        else:
            st.caption("Process transcripts first.")


if process_button_clicked:
    with st.status(f"Processing '{final_transcript_name_for_processing}' (embedding chunks)...", expanded=True) as status_container:
        st.write("Checking for existing transcript..."); logger.info(f"Processing: '{final_transcript_name_for_processing}'")
        already_exists = False
        try:
            scroll_res, _ = qdrant_client.scroll(COLLECTION_NAME, scroll_filter=models.Filter(must=[models.FieldCondition(key="transcript_name",match=models.MatchValue(value=final_transcript_name_for_processing))]), limit=1, with_payload=False, with_vectors=False)
            already_exists = bool(scroll_res)
        except Exception as e: status_container.update(label=f"Qdrant check error: {e}",state="error"); logger.error(f"Qdrant check error: {e}"); st.stop()
        
        if already_exists:
            st.write(f"⚠️ Transcript '{final_transcript_name_for_processing}' may already exist. Processing will continue.")
            status_container.update(label=f"Processing '{final_transcript_name_for_processing}' (Note: may already exist)...", state="running")

        temp_srt_path = ""
        try:
            st.write(f"Saving temp SRT for '{uploaded_file.name}'...");
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".srt", delete=False) as tmp_f: tmp_f.write(uploaded_file.getvalue()); temp_srt_path = tmp_f.name
            st.write("Parsing SRT..."); parsed_subs = parse_srt_file(temp_srt_path)
            if not parsed_subs: status_container.update(label="SRT empty/unparsable.",state="error"); st.stop()
            st.write(f"Chunking (size: {chunk_size_words_input}, overlap: {overlap_words_input})...");
            chunks = srt_to_chunks(parsed_subs, chunk_size_words_input, overlap_words_input); total_c = len(chunks)
            if total_c == 0: status_container.update(label="No chunks created.",state="error"); st.stop()
            st.write(f"{total_c} chunks created.")

            prog = st.progress(0, text="Embedding & Storing Chunks...")
            all_q_points = []
            for i in range(0, total_c, EMBED_BATCH_SIZE):
                batch_chunks = chunks[i : i + EMBED_BATCH_SIZE]
                texts = [c['text'] for c in batch_chunks]
                vecs = batch_get_embeddings(texts)
                batch_q_points = []
                for idx, (chk_data, vec_emb) in enumerate(zip(batch_chunks, vecs)):
                    actual_chunk_idx = i + idx
                    prog_text_update = f"Embedding & Storing: {actual_chunk_idx+1}/{total_c}"
                    if vec_emb is None: prog.progress((actual_chunk_idx+1)/total_c, text=f"{prog_text_update} (Skipped)"); continue
                    payload_dict = {"transcript_name": final_transcript_name_for_processing, "timestamp": chk_data['timestamp'], "original_text": chk_data['text']}
                    batch_q_points.append({'id': str(uuid.uuid4()), 'vector': vec_emb, 'payload': payload_dict})
                    prog.progress((actual_chunk_idx+1)/total_c, text=prog_text_update)
                if batch_q_points: all_q_points.extend(batch_q_points)
            
            if all_q_points: st.write(f"Upserting {len(all_q_points)} points to Qdrant..."); upsert_to_qdrant(all_q_points)
            status_container.update(label=f"'{final_transcript_name_for_processing}' processed and stored!", state="complete")
            logger.info(f"Stored transcript: {final_transcript_name_for_processing}"); get_processed_transcripts.clear()
        except Exception as e_proc: status_container.update(label=f"Processing error: {e_proc}",state="error"); logger.error(f"Processing error: {e_proc}", exc_info=True)
        finally:
            if temp_srt_path and Path(temp_srt_path).exists():
                try: Path(temp_srt_path).unlink(); logger.info(f"Deleted temp SRT: {temp_srt_path}")
                except OSError as e_del_f: logger.error(f"Error deleting temp file {temp_srt_path}: {e_del_f}")

# --- Search UI ---
st.markdown("---")
st.header("🔍 Semantic Search & Analysis")
search_query_input = st.text_input("Enter your search query:", key="main_search_query_input")

st.write("**Filter by Biographical Content** (Applies if fine-tuned bio-extraction has been run):")
human_readable_bio_categories = {key: key.replace("_", " ").title() for key in BIOGRAPHICAL_CATEGORY_KEYS}
selected_bio_categories_human_readable = st.multiselect(
    "Filter for chunks containing ANY of these biographical aspects:",
    options=list(human_readable_bio_categories.values()),
    key="bio_category_multiselect_filter_key"
)
selected_bio_category_keys_for_qdrant_filter = [
    key for key, human_readable in human_readable_bio_categories.items()
    if human_readable in selected_bio_categories_human_readable
]

col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1: use_llm_reranking = st.checkbox("Enable LLM Reranking", value=True, key="rerank_toggle_checkbox", help=f"Uses {RERANK_MODEL}")
with col_opt2: do_pinpoint_answer_extraction = st.checkbox("Pinpoint Answer Snippet", value=True, key="pinpoint_answer_checkbox", help=f"Uses {ANSWER_EXTRACTION_MODEL}")
with col_opt3: initial_search_results_limit = st.slider("Initial results:", 3, 30, 5, key="search_results_limit_slider")
custom_reranking_instructions_input = ""
if use_llm_reranking: custom_reranking_instructions_input = st.text_area("Custom Reranking Instructions (Optional):", placeholder="e.g., 'Prioritize practical advice.'", key="custom_rerank_instructions_input_area", height=100)

if search_query_input:
    with st.spinner("Searching and analyzing results..."):
        try: query_vector = client.embeddings.create(model=EMBEDDING_MODEL, input=[search_query_input]).data[0].embedding
        except Exception as e: st.error(f"Query embed fail: {e}"); logger.error(f"Query embed error: {e}"); st.stop()
        
        qdrant_filter_conditions = []
        if selected_bio_category_keys_for_qdrant_filter:
            for cat_key in selected_bio_category_keys_for_qdrant_filter:
                flag_field_name = f"has_{cat_key}"
                qdrant_filter_conditions.append(models.FieldCondition(key=flag_field_name, match=models.MatchValue(value=True)))
        
        final_qdrant_filter = models.Filter(should=qdrant_filter_conditions) if qdrant_filter_conditions else None

        try: 
            qdrant_hits = qdrant_client.search(
                collection_name=COLLECTION_NAME, query_vector=query_vector, 
                query_filter=final_qdrant_filter, limit=initial_search_results_limit, with_payload=True
            )
        except Exception as e: st.error(f"Qdrant search fail: {e}"); logger.error(f"Qdrant search error: {e}"); st.stop()
        
        retrieved_payloads = [h.payload for h in qdrant_hits if h.payload]
        if not retrieved_payloads: st.info("No results found matching your query and selected filters."); st.stop()

        if use_llm_reranking and len(retrieved_payloads) > 1:
            with st.spinner(f"Reranking with {RERANK_MODEL}..."):
                excerpts = [f"[{i+1}] {rp.get('original_text', 'N/A')}" for i, rp in enumerate(retrieved_payloads)]
                instr = "Rerank by relevance to query."
                if custom_reranking_instructions_input: instr += f" Criteria: {custom_reranking_instructions_input}."
                prompt_rerank = (f"{instr}\nQuery: \"{search_query_input}\"\nExcerpts:\n{chr(10).join(excerpts)}\nReturn only comma-separated original indices reranked.")
                try:
                    resp = client.chat.completions.create(model=RERANK_MODEL, messages=[{"role":"system", "content":"You rerank results."}, {"role":"user", "content":prompt_rerank}], max_tokens=200, temperature=0.0)
                    indices_str = resp.choices[0].message.content.strip()
                    parsed_idx, seen_idx = [], set()
                    for s_idx in re.findall(r'\d+', indices_str):
                        val = int(s_idx) - 1
                        if 0 <= val < len(retrieved_payloads) and val not in seen_idx: parsed_idx.append(val); seen_idx.add(val)
                    if parsed_idx:
                        final_order_idx = parsed_idx + [i for i in range(len(retrieved_payloads)) if i not in seen_idx]
                        retrieved_payloads = [retrieved_payloads[i] for i in final_order_idx]
                        st.caption(f"ℹ️ Results reranked by {RERANK_MODEL}.")
                    else: st.warning("Reranking failed/no change. Original order.", icon="⚠️")
                except Exception as e_rr_exc: st.warning(f"Reranking error: {e_rr_exc}. Original order.", icon="⚠️"); logger.error(f"Reranking error: {e_rr_exc}")
        
        st.subheader(f"Search Results for: \"{search_query_input}\"")
        if selected_bio_category_keys_for_qdrant_filter:
            st.caption(f"Filtered for: {', '.join(selected_bio_categories_human_readable)}")

        for i, payload_data_item in enumerate(retrieved_payloads, 1):
            if not isinstance(payload_data_item, dict): continue
            full_chunk_text_item = payload_data_item.get('original_text', 'Error: Text missing')
            exp_title_str = f"Result {i} @ {payload_data_item.get('timestamp','N/A')} | Transcript: {payload_data_item.get('transcript_name','N/A')}"
            
            with st.expander(exp_title_str, expanded=(i==1)):
                if do_pinpoint_answer_extraction and full_chunk_text_item != 'Error: Text missing':
                    with st.spinner(f"Pinpointing answer for result {i}..."):
                        answer_span_text = extract_answer_span(search_query_input, full_chunk_text_item, ANSWER_EXTRACTION_MODEL)
                    if answer_span_text and answer_span_text.strip() and \
                       answer_span_text.strip().lower() != full_chunk_text_item.strip().lower() and \
                       len(answer_span_text.strip()) < len(full_chunk_text_item.strip()):
                        st.markdown(f"🎯 **Pinpointed Answer Snippet:**\n> *{answer_span_text.strip()}*"); st.markdown("--- \n**Full Context Chunk:**")
                    elif answer_span_text and answer_span_text.strip().lower() == full_chunk_text_item.strip().lower():
                         st.caption(f"(Full chunk identified as most direct answer by {ANSWER_EXTRACTION_MODEL})")
                
                st.markdown(full_chunk_text_item)
                
                claude_entities_dict = payload_data_item.get('entities')
                if claude_entities_dict and isinstance(claude_entities_dict, dict):
                    entity_display_parts = []
                    if claude_entities_dict.get('people'): entity_display_parts.append(f"**People:** {', '.join(claude_entities_dict['people'])}")
                    if claude_entities_dict.get('places'): entity_display_parts.append(f"**Places:** {', '.join(claude_entities_dict['places'])}")
                    if 'self_references' in claude_entities_dict:
                        sr_val = claude_entities_dict.get('self_references')
                        sr_disp = "Yes" if sr_val is True else ("No" if sr_val is False else str(sr_val))
                        entity_display_parts.append(f"**Self-ref:** {sr_disp}")
                    if entity_display_parts: st.markdown("---"); st.markdown(" ".join(entity_display_parts), unsafe_allow_html=True)
                
                fine_tuned_bio_data = payload_data_item.get('biographical_extractions')
                if fine_tuned_bio_data and isinstance(fine_tuned_bio_data, dict):
                    st.markdown("--- \n**Biographical Details (Extracted by Fine-tuned Model):**")
                    has_any_bio_detail_displayed = False
                    for bio_key in BIOGRAPHICAL_CATEGORY_KEYS:
                        if bio_key in fine_tuned_bio_data and fine_tuned_bio_data[bio_key]: # Check if list is not empty
                            has_any_bio_detail_displayed = True
                            display_key_name = bio_key.replace("_", " ").title()
                            st.markdown(f"**{display_key_name}:**")
                            for item_quote in fine_tuned_bio_data[bio_key]:
                                st.markdown(f"- *\"{item_quote}\"*")
                    if not has_any_bio_detail_displayed:
                        st.caption("(No specific biographical details extracted by the fine-tuned model for the defined categories in this chunk.)")

# === Satsang Data Explorer UI ===
st.markdown("---")
st.header("🔬 Satsang Data Explorer")
st.info("Use this section to inspect the processed data for a specific transcript. Select a transcript from the dropdown to view its chunks and the associated tags and categories.")

# Function to fetch all chunks for a given transcript name, with caching
@st.cache_data(ttl=600)
def get_chunks_for_transcript(transcript_name: str) -> list:
    """Fetches all points (chunks) for a specific transcript from Qdrant."""
    if not qdrant_client or not transcript_name:
        return []
    
    logger.info(f"Fetching all chunks from Qdrant for transcript: '{transcript_name}'")
    all_points = []
    try:
        offset_val = None
        while True:
            points_batch, next_offset_val = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="transcript_name", match=models.MatchValue(value=transcript_name))
                ]),
                limit=100, # Fetch in batches of 100
                offset=offset_val,
                with_payload=True,
                with_vectors=False
            )
            if points_batch:
                all_points.extend(points_batch)
            
            if not next_offset_val:
                break # No more pages
            offset_val = next_offset_val
            
        logger.info(f"Retrieved {len(all_points)} chunks for '{transcript_name}'.")
        # Sort by timestamp to ensure chronological order
        all_points.sort(key=lambda p: p.payload.get('timestamp', '0:0:0.0'))
        return all_points
    except Exception as e:
        st.error(f"Failed to fetch chunks for '{transcript_name}': {e}")
        logger.error(f"Error in get_chunks_for_transcript for '{transcript_name}': {e}", exc_info=True)
        return []

# Dropdown to select the transcript to inspect
processed_transcript_list_for_explorer = get_processed_transcripts()
if processed_transcript_list_for_explorer:
    selected_transcript_to_inspect = st.selectbox(
        "Select a transcript to inspect:",
        options=[""] + processed_transcript_list_for_explorer,
        index=0,
        key="transcript_inspector_selectbox"
    )

    if selected_transcript_to_inspect:
        with st.spinner(f"Loading all chunks for '{selected_transcript_to_inspect}'..."):
            transcript_chunks = get_chunks_for_transcript(selected_transcript_to_inspect)

        if not transcript_chunks:
            st.warning("No chunks found for this transcript, or an error occurred during fetching.")
        else:
            st.success(f"Loaded {len(transcript_chunks)} chunks. Expand any chunk to see details.")
            
            for i, point in enumerate(transcript_chunks):
                payload = point.payload
                if not isinstance(payload, dict): continue

                exp_title = f"Chunk {i+1}  |  Timestamp: {payload.get('timestamp', 'N/A')}"
                with st.expander(exp_title):
                    
                    # --- Display Original Text ---
                    st.markdown("**Full Text:**")
                    st.markdown(f"> {payload.get('original_text', 'Error: Text missing')}")
                    
                    st.markdown("---")
                    
                    # --- Display Claude Entities ---
                    st.subheader("🏷️ Claude Entity Tags")
                    claude_entities = payload.get('entities')
                    if claude_entities and isinstance(claude_entities, dict):
                        st.json(claude_entities)
                    else:
                        st.caption("No Claude entity tags found for this chunk.")
                        
                    st.markdown("---")

                    # --- Display Fine-Tuned Model Categories ---
                    st.subheader("🗂️ Fine-Tuned Model Categories")
                    # Find all assigned categories by checking the 'has_' flags
                    assigned_categories = [
                        key.replace("_", " ").title() 
                        for key in BIOGRAPHICAL_CATEGORY_KEYS 
                        if payload.get(f"has_{key}") is True
                    ]
                    
                    if assigned_categories:
                        st.markdown("**Assigned Categories:**")
                        for cat in assigned_categories:
                            st.markdown(f"- {cat}")
                    else:
                        st.caption("No biographical categories were assigned by the fine-tuned model for this chunk.")
                    
                    # Also display the raw extracted quotes for verification
                    fine_tuned_extractions = payload.get('biographical_extractions')
                    if fine_tuned_extractions and isinstance(fine_tuned_extractions, dict):
                        st.markdown("**Extracted Quotes (for verification):**")
                        has_quotes = False
                        for key, quotes in fine_tuned_extractions.items():
                            if quotes and isinstance(quotes, list):
                                has_quotes = True
                                st.markdown(f"- **{key.replace('_', ' ').title()}:** {quotes}")
                        if not has_quotes:
                             st.caption("(No specific quotes were extracted.)")
                    
else:
    st.caption("Process a transcript first to inspect its data here.")



