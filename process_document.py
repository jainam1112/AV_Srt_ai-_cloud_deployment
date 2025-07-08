import json
import openai
import time
from docx import Document
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, ValidationError, Field

# Define schema using pydantic
class TimelineItem(BaseModel):
    UID: str
    StandardizedDate: str
    DateFootnote: str = ""
    Summary: str
    ThematicCategories: List[str]
    AssociatedPeople: List[str]
    AssociatedPlaces: List[str]
    FormattedSnippet: str
    SourceDocument: str
    SourcePage: int

class SemanticItem(BaseModel):
    Content: str
    Metadata: Dict[str, Any]

class GlossaryItem(BaseModel):
    Term: str
    Definition: str

class ArchiveData(BaseModel):
    TimelineData: List[TimelineItem]
    SemanticSearchIndexData: List[SemanticItem]
    GlossaryData: List[GlossaryItem]
    ProcessingMetadata: Dict[str, Any] = Field(default_factory=dict)

# Chunking
def chunk_text(text, max_chars=10000):
    paragraphs = text.split("\n")
    chunks = []
    chunk = ""
    for para in paragraphs:
        if len(chunk) + len(para) < max_chars:
            chunk += para + "\n"
        else:
            chunks.append(chunk.strip())
            chunk = para + "\n"
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# OpenAI API with retry
def safe_openai_call(client, messages, model="gpt-4-1106-preview", retries=3):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2
            )
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

def process_docx_with_openai(docx_file, api_key, model="gpt-4-1106-preview"):
    client = openai.OpenAI(api_key=api_key)
    doc = Document(docx_file)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    if not text:
        raise ValueError("The uploaded document appears to be empty.")

    prompt_path = Path("system_prompt.txt")
    if not prompt_path.exists():
        raise FileNotFoundError("system_prompt.txt is missing.")
    prompt = prompt_path.read_text()

    chunks = chunk_text(text)
    cumulative_output = {
        "TimelineData": [],
        "SemanticSearchIndexData": [],
        "GlossaryData": []
    }

    for i, chunk in enumerate(chunks):
        user_msg = f"Document Name: {docx_file.name}\n\n--- DOCUMENT CHUNK ({i+1}/{len(chunks)}) ---\n\n{chunk}"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg}
        ]
        response = safe_openai_call(client, messages, model=model)
        parsed = json.loads(response.choices[0].message.content)

        cumulative_output["TimelineData"].extend(parsed.get("TimelineData", []))
        cumulative_output["SemanticSearchIndexData"].extend(parsed.get("SemanticSearchIndexData", []))

        existing_terms = {g["Term"] for g in cumulative_output["GlossaryData"]}
        for glossary in parsed.get("GlossaryData", []):
            if glossary["Term"] not in existing_terms:
                cumulative_output["GlossaryData"].append(glossary)

    cumulative_output["ProcessingMetadata"] = {
        "DocumentName": docx_file.name,
        "PromptVersion": "v1.0",
        "ProcessedOn": datetime.utcnow().isoformat() + "Z",
        "ModelUsed": model
    }

    try:
        validated = ArchiveData(**cumulative_output)
        return json.loads(validated.json(indent=2))
    except ValidationError as ve:
        raise ValueError("Validation error in LLM output: " + str(ve))
