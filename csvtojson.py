import pandas as pd
import json

# Load your CSV
df = pd.read_csv("C:\\Users\\jaina\\Downloads\\subtitles\\Copy of Data Extraction Output - Copy of Missing categories.csv")  # Replace with your file path

# Optional: Strip whitespace
df['chunk'] = df['chunk'].astype(str).str.strip()
df['annotated'] = df['annotated'].astype(str).str.strip()
df['category'] = df['category'].astype(str).str.strip()

# Group by chunk for consolidation (if needed)
jsonl_lines = []

for chunk_text, group in df.groupby('chunk'):
    assistant_data = {}
    for _, row in group.iterrows():
        category = row['category']
        quotes = [q.strip() for q in row['annotated'].split("%%%QUOTE_SEPARATOR%%%") if q.strip()]
        if category in assistant_data:
            assistant_data[category].extend(quotes)
        else:
            assistant_data[category] = quotes

    jsonl_obj = {
        "messages": [
            {"role": "user", "content": f'Transcript Segment: "{chunk_text}"'},
            {"role": "assistant", "content": json.dumps(assistant_data, ensure_ascii=False)}
        ]
    }
    jsonl_lines.append(jsonl_obj)

# Save to .jsonl
output_path = "C:\\Users\\jaina\\satsang_search_app\\cloud_deployment\\newtrain.json"
with open(output_path, "w", encoding="utf-8") as f:
    for line in jsonl_lines:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print(f"✅ JSONL training file saved to: {output_path}")
