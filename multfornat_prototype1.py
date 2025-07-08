import streamlit as st
import os
import uuid
import srt
from pathlib import Path
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qdrant_models
from elasticsearch import Elasticsearch, exceptions as es_exceptions
import spacy # For basic NER

# --- Configuration ---
load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = "guru_archive_prototype_qdrant"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # New line for QDRANT_API_KEY

# Elasticsearch
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_INDEX_NAME = "guru_archive_prototype" # Should match the index created by es_setup.py

# spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None # Will disable NER features

# --- Initialize Clients ---
@st.cache_resource
def get_openai_client():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found.")
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_qdrant_client():
    try:
        qdrant_params = {"host": QDRANT_HOST, "port": QDRANT_PORT}
        if QDRANT_API_KEY:
            qdrant_params["api_key"] = QDRANT_API_KEY
            if not QDRANT_HOST.startswith("http") and QDRANT_HOST != "localhost":
                qdrant_params["https"] = True
        client = QdrantClient(**qdrant_params)
        # Ensure collection exists
        try:
            client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        except Exception: # Simplified: if get fails, try to create/recreate
            client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(size=EMBEDDING_DIM, distance=qdrant_models.Distance.COSINE)
            )
            # Create payload index for filtering (if needed later)
            # client.create_payload_index(collection_name=QDRANT_COLLECTION_NAME, field_name="source_type", field_schema="keyword")
        return client
    except Exception as e:
        qdrant_error = f"Failed to connect to Qdrant: {e}. Please check connection details and ensure Qdrant is running."
        st.error(qdrant_error)
        return None

@st.cache_resource
def get_es_client():
    try:
        if ES_USER and ES_PASSWORD:
            client = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASSWORD), request_timeout=30)
        else:
            client = Elasticsearch(ES_HOST, request_timeout=30)
        if not client.ping():
            raise ConnectionError("ES Ping failed")
        return client
    except Exception as e:
        st.error(f"Failed to connect to Elasticsearch: {e}")
        return None

openai_client = get_openai_client()
qdrant_client = get_qdrant_client()
es_client = get_es_client()

# --- Helper Functions ---
def parse_srt_file(file_content_bytes: bytes) -> list:
    try:
        file_content_str = file_content_bytes.decode('utf-8')
        return list(srt.parse(file_content_str))
    except Exception as e:
        st.error(f"Error parsing SRT file: {e}")
        return []

def srt_to_chunks(subs: list, chunk_size_words: int = 300, overlap_words: int = 50) -> list:
    chunks, current_sub_idx = [], 0
    if not subs: return []
    while current_sub_idx < len(subs):
        buffer, word_count, temp_idx = [], 0, current_sub_idx
        while temp_idx < len(subs):
            sub = subs[temp_idx]
            words_in_sub = len(sub.content.split())
            if not buffer or word_count + words_in_sub <= chunk_size_words + (chunk_size_words * 0.2):
                buffer.append(sub); word_count += words_in_sub
            else: break
            temp_idx += 1
        if not buffer: break
        chunk_text = ' '.join(s.content for s in buffer)
        chunks.append({
            'text': chunk_text,
            'timestamp_start_str': str(buffer[0].start),
            'timestamp_end_str': str(buffer[-1].end) # For context
        })
        if overlap_words <= 0 or len(buffer) <= 1: current_sub_idx = temp_idx
        else:
            # Simplified overlap, advance by non-overlapping part
            # A more robust overlap would count back words from end of buffer
            subs_to_advance = max(1, len(buffer) - int(len(buffer) * (overlap_words / word_count if word_count > 0 else 0.1)))
            current_sub_idx += subs_to_advance
            if current_sub_idx >= temp_idx : current_sub_idx = temp_idx # ensure progress if calculation is off
    return chunks

def get_embeddings(texts: list) -> list:
    if not openai_client or not texts: return [None] * len(texts)
    try:
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [obj.embedding for obj in response.data]
    except OpenAIError as e:
        st.warning(f"OpenAI embedding error: {e}")
        return [None] * len(texts)

def extract_people_entities_spacy(text: str) -> list:
    if not nlp: return []
    doc = nlp(text)
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    # Simple heuristic for "Sri X", "Yji" if not caught by PERSON directly
    # This is very basic and might need refinement
    additional_people = []
    import re
    for token in text.split(): # very naive
        if token.startswith("Sri") and len(token) > 3:
            additional_people.append(token)
        if token.endswith("ji") and len(token) > 2:
            additional_people.append(token)
    return list(set(people + additional_people))


# --- Streamlit UI ---
st.set_page_config(page_title="Guru Archive Prototype", layout="wide")
st.title("🕉️ Guru's Life Archive - Prototype")

# --- Ingestion Section ---
st.sidebar.header("1. Ingest Discourse (SRT)")
uploaded_file = st.sidebar.file_uploader("Upload SRT file", type=["srt"])

if uploaded_file and openai_client and qdrant_client and es_client:
    if st.sidebar.button("Process and Ingest File"):
        discourse_title = Path(uploaded_file.name).stem
        with st.spinner(f"Processing '{discourse_title}'..."):
            srt_subs = parse_srt_file(uploaded_file.getvalue())
            if not srt_subs:
                st.error("Could not parse SRT or file is empty.")
                st.stop()

            raw_chunks = srt_to_chunks(srt_subs)
            if not raw_chunks:
                st.error("No chunks created from SRT.")
                st.stop()
            
            st.sidebar.success(f"Created {len(raw_chunks)} chunks.")

            # Prepare for batching
            qdrant_points = []
            es_bulk_data = []
            
            chunk_texts_for_embedding = [chunk['text'] for chunk in raw_chunks]
            embeddings = get_embeddings(chunk_texts_for_embedding)

            for i, chunk_data in enumerate(raw_chunks):
                if embeddings[i] is None:
                    st.warning(f"Skipping chunk {i+1} due to embedding failure.")
                    continue

                chunk_id = str(uuid.uuid4())
                mentioned_people = extract_people_entities_spacy(chunk_data['text'])

                # Data for Qdrant and Elasticsearch
                payload = {
                    "chunk_id": chunk_id, # Also store in payload for easier retrieval by ID from ES results
                    "original_text": chunk_data['text'],
                    "source_type": "discourse_gurudev",
                    "discourse_title": discourse_title,
                    "timestamp_start_str": chunk_data['timestamp_start_str'],
                    "mentioned_people": mentioned_people
                }

                # Qdrant point
                qdrant_points.append(qdrant_models.PointStruct(
                    id=chunk_id,
                    vector=embeddings[i],
                    payload=payload
                ))

                # Elasticsearch document
                es_bulk_data.append({"index": {"_index": ES_INDEX_NAME, "_id": chunk_id}})
                es_bulk_data.append(payload)

            # Batch upsert to Qdrant
            if qdrant_points:
                try:
                    qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=qdrant_points)
                    st.sidebar.success(f"Upserted {len(qdrant_points)} chunks to Qdrant.")
                except Exception as e:
                    st.sidebar.error(f"Qdrant upsert error: {e}")
            
            # Batch index to Elasticsearch
            if es_bulk_data:
                try:
                    from elasticsearch.helpers import bulk
                    successes, errors = bulk(es_client, es_bulk_data, raise_on_error=False)
                    st.sidebar.success(f"Indexed {successes} chunks to Elasticsearch.")
                    if errors:
                        st.sidebar.warning(f"ES Bulk Errors: {len(errors)}")
                        # for error in errors[:3]: st.sidebar.caption(str(error)) # Show first few errors
                except es_exceptions.ElasticsearchException as e:
                    st.sidebar.error(f"Elasticsearch bulk index error: {e}")
        st.sidebar.info("Processing complete.")

# --- Search Section ---
st.header("2. Hybrid Search Archive")
query = st.text_input("Enter search query (names, keywords, concepts):")

col1, col2, col3 = st.columns(3)
semantic_weight = col1.slider("Semantic Weight:", 0.0, 1.0, 0.7, 0.05)
lexical_weight = 1.0 - semantic_weight # Ensure weights sum to 1
col2.metric("Lexical Weight", f"{lexical_weight:.2f}") # Display lexical weight
top_k_results = col3.number_input("Number of results (K):", 1, 20, 5)


if query and openai_client and qdrant_client and es_client:
    with st.spinner("Searching..."):
        # 1. Semantic Search (Qdrant)
        qdrant_results = []
        try:
            query_embedding = get_embeddings([query])[0]
            if query_embedding:
                hits = qdrant_client.search(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=top_k_results * 2, # Fetch more to allow for diverse merging
                    with_payload=True 
                )
                for hit in hits:
                    qdrant_results.append({
                        "id": hit.id, 
                        "score": hit.score, 
                        "payload": hit.payload,
                        "search_type": "semantic"
                    })
        except Exception as e:
            st.error(f"Qdrant search error: {e}")

        # 2. Lexical Search (Elasticsearch)
        es_results = []
        try:
            es_query_body = {
                "query": {
                    "multi_match": { # Simple multi_match for prototype
                        "query": query,
                        "fields": ["original_text", "discourse_title", "mentioned_people"] # Search these fields
                    }
                }
            }
            response = es_client.search(index=ES_INDEX_NAME, body=es_query_body, size=top_k_results * 2)
            for hit in response['hits']['hits']:
                es_results.append({
                    "id": hit["_id"], 
                    "score": hit["_score"], 
                    "payload": hit["_source"],
                    "search_type": "lexical"
                })
        except es_exceptions.ElasticsearchException as e:
            st.error(f"Elasticsearch search error: {e}")

        # 3. Combine and Re-rank (Simple Weighted Sum for Prototype - RRF is better for production)
        # For simplicity, we'll normalize scores (0-1) then apply weights
        # This is a VERY basic normalization and ranking.
        
        # Normalize ES scores (max score can vary greatly)
        max_es_score = max(r['score'] for r in es_results) if es_results else 1
        for r in es_results:
            r['normalized_score'] = (r['score'] / max_es_score) if max_es_score > 0 else 0
            
        # Qdrant cosine similarity is already 0-1 (higher is better)
        for r in qdrant_results:
            r['normalized_score'] = r['score'] 

        # Combine
        combined_by_id = {}
        for r_list in [qdrant_results, es_results]:
            for r in r_list:
                doc_id = r['id']
                if doc_id not in combined_by_id:
                    combined_by_id[doc_id] = {'payload': r['payload'], 'semantic_score': 0, 'lexical_score': 0, 'sources': []}
                
                if r['search_type'] == 'semantic':
                    combined_by_id[doc_id]['semantic_score'] = r['normalized_score']
                else: # lexical
                    combined_by_id[doc_id]['lexical_score'] = r['normalized_score']
                combined_by_id[doc_id]['sources'].append(r['search_type'])

        final_ranked_results = []
        for doc_id, scores_data in combined_by_id.items():
            final_score = (scores_data['semantic_score'] * semantic_weight) + \
                          (scores_data['lexical_score'] * lexical_weight)
            final_ranked_results.append({
                "id": doc_id,
                "score": final_score,
                "payload": scores_data['payload'],
                "sources": list(set(scores_data['sources'])) # Unique sources
            })
            
        final_ranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Display top K results
        st.subheader(f"Top {top_k_results} Hybrid Search Results for '{query}':")
        if not final_ranked_results:
            st.info("No results found.")
        else:
            for i, result in enumerate(final_ranked_results[:top_k_results]):
                payload = result['payload']
                st.markdown(f"--- \n **{i+1}. Score: {result['score']:.4f} (Found via: {', '.join(result['sources'])})**")
                st.markdown(f"**Title:** {payload.get('discourse_title', 'N/A')} | **Timestamp:** {payload.get('timestamp_start_str', 'N/A')}")
                st.markdown(f"> {payload.get('original_text')}")
                
                mentioned_people = payload.get('mentioned_people', [])
                if mentioned_people:
                    st.markdown(f"**Mentioned People (Graph Link Prototype):**")
                    for person in mentioned_people:
                        if st.button(person, key=f"person_{result['id']}_{person.replace(' ','_')}", help=f"Explore connections for {person} (not implemented in proto)"):
                            st.info(f"Prototype: Would explore graph connections for {person}.") # Placeholder
                st.caption(f"Chunk ID: {result['id']}")

elif query:
    st.warning("One or more clients (OpenAI, Qdrant, Elasticsearch) are not initialized. Cannot search.")