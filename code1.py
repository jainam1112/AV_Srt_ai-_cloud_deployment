
# --- Category Annotation Inputs ---
st.header("2. Extract Verbatim Quotes for Each Category")
st.markdown(f"""
**Instructions:**
- For each category below, copy and paste the **EXACT verbatim quote(s)** from the transcript chunk above that fit the category.
- **To preserve newlines *within* a single quote**, just paste it as is.
- If **multiple quotes** fit into one category, separate them using the line: `{QUOTE_SEPARATOR}`

**Example:**
```text
This is the first quote.
It may span multiple lines.
{QUOTE_SEPARATOR}
This is the second quote for the same category.
```
""")

cols_per_row = 2
annotation_inputs = {}
category_columns = st.columns(cols_per_row)

for i, key in enumerate(ALL_BIOGRAPHICAL_CATEGORY_KEYS):
    col = category_columns[i % cols_per_row]
    with col:
        label = f"{i + 1}. {format_key_for_display(key)}"
        current_val = st.session_state.current_annotations.get(key, "")
        input_val = st.text_area(
            label,
            height=120,
            key=f"category_textarea_{key}",
            placeholder=f"Quotes for {format_key_for_display(key)}...",
            value=current_val
        )
        if input_val != current_val:
            st.session_state.current_annotations[key] = input_val
        annotation_inputs[key] = st.session_state.current_annotations[key]

# --- JSONL Generator ---
st.header("3. Generate Fine-Tuning Example (JSONL Format)")

if st.button("Generate JSONL Line", key="generate_button_main"):
    current_chunk = st.session_state.transcript_chunk_input.strip()

    if not current_chunk:
        st.warning("Please load or paste a transcript chunk before generating.")
    else:
        assistant_json_data = {}
        for key, raw_text in st.session_state.current_annotations.items():
            if raw_text.strip():
                split_quotes = [q.strip() for q in raw_text.split(QUOTE_SEPARATOR) if q.strip()]
                assistant_json_data[key] = split_quotes
            else:
                assistant_json_data[key] = []

        assistant_content_str = json.dumps(assistant_json_data, indent=2)
        user_message = f'Transcript Segment: "{current_chunk}"'
        jsonl_line = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"```json\n{assistant_content_str}\n```"}
            ]
        }

        st.subheader("Generated JSONL Line (Copy this):")
        st.code(json.dumps(jsonl_line), language="json")
        st.info("Copy this line into your `.jsonl` file. Each line should be a self-contained training example.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Annotations (Keep Chunk)", key="clear_annotations_only_button"):
                st.session_state.current_annotations = {key: "" for key in ALL_BIOGRAPHICAL_CATEGORY_KEYS}
                st.rerun()

        with col2:
            if st.button("Clear All (Chunk & Annotations)", key="clear_all_button"):
                st.session_state.transcript_chunk_input = ""
                st.session_state.loaded_qdrant_chunk_id = None
                st.session_state.loaded_qdrant_chunk_text = ""
                st.session_state.current_annotations = {key: "" for key in ALL_BIOGRAPHICAL_CATEGORY_KEYS}
                st.rerun()

st.markdown("---")
st.markdown("Ensure consistency and accuracy in your annotations for the best fine-tuning results!")
