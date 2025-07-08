import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np

def render_semantic_search(semantic_data):
    st.subheader("🔍 Semantic Search")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = [item["Content"] for item in semantic_data]
    embeddings = model.encode(corpus, convert_to_tensor=True)

    query = st.text_input("Enter your search query")
    if query:
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
        top_indices = np.argsort(-scores)[:5]

        for idx in top_indices:
            item = semantic_data[idx]
            st.markdown(f"**Score:** {scores[idx]:.2f}")
            st.markdown(item["Content"])
            st.caption(f"Source UID: {item['Metadata']['UID']}, Themes: {', '.join(item['Metadata']['Themes'])}")
