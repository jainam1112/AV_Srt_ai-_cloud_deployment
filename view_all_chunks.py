#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv, find_dotenv
import pandas as pd # Added for Excel export

# === CONFIG ===
load_dotenv(find_dotenv()) # Load .env file if present

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # For Qdrant Cloud
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gurudev_satsangs") # Ensure this matches your collection
OUTPUT_EXCEL_FILE = "qdrant_chunks_export.xlsx" # Name of the output Excel file

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_all_chunks(
    qdrant_client: QdrantClient,
    collection_name: str,
    limit_per_scroll: int = 100,
    max_chunks_to_fetch: int = None # Set to None to fetch all, or a number to limit
):
    """
    Fetches chunks from the specified Qdrant collection using the scroll API.

    Args:
        qdrant_client: Initialized QdrantClient instance.
        collection_name: Name of the collection to fetch from.
        limit_per_scroll: Number of points to fetch in each scroll request.
        max_chunks_to_fetch: Maximum number of chunks to fetch in total.
                             If None, fetches all chunks.

    Returns:
        A list of PointStruct objects.
    """
    all_points = []
    current_offset = None
    fetched_count = 0

    logger.info(f"Starting to fetch chunks from collection '{collection_name}'...")

    while True:
        try:
            logger.debug(f"Scrolling with offset: {current_offset}, limit: {limit_per_scroll}")
            points, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                offset=current_offset,
                limit=limit_per_scroll,
                with_payload=True,  # We need the payload to see the chunk data
                with_vectors=False  # Set to True if you also need the vectors
            )
        except Exception as e:
            logger.error(f"Error during Qdrant scroll: {e}")
            break

        if not points:
            logger.info("No more points found in the collection.")
            break

        all_points.extend(points)
        fetched_count += len(points)
        logger.info(f"Fetched {len(points)} points in this batch. Total fetched: {fetched_count}")

        if max_chunks_to_fetch is not None and fetched_count >= max_chunks_to_fetch:
            logger.info(f"Reached max_chunks_to_fetch limit of {max_chunks_to_fetch}.")
            # Truncate if we overshot due to batch size
            all_points = all_points[:max_chunks_to_fetch]
            break

        if next_offset is None:
            logger.info("Reached the end of the collection.")
            break
        current_offset = next_offset

    logger.info(f"Finished fetching. Total chunks retrieved: {len(all_points)}")
    return all_points

def export_chunks_to_excel(points: list, filename: str):
    """
    Exports the fetched Qdrant points (chunks) to an Excel file.

    Args:
        points: A list of PointStruct objects.
        filename: The name of the Excel file to create.
    """
    if not points:
        logger.info("No points to export to Excel.")
        return

    data_for_excel = []
    for point in points:
        row = {'id': str(point.id)} # Ensure ID is string for consistency
        if point.payload:
            row['transcript_name'] = point.payload.get('transcript_name')
            row['timestamp'] = point.payload.get('timestamp')
            row['original_text'] = point.payload.get('original_text')
            # Handle entities if they exist and are dicts; otherwise, store as string or None
            entities = point.payload.get('entities')
            if isinstance(entities, dict):
                row['entities_people'] = ", ".join(entities.get('people', [])) if entities.get('people') else None
                row['entities_places'] = ", ".join(entities.get('places', [])) if entities.get('places') else None
                row['entities_self_references'] = entities.get('self_references')
            else:
                row['entities_people'] = None
                row['entities_places'] = None
                row['entities_self_references'] = None
        data_for_excel.append(row)

    df = pd.DataFrame(data_for_excel)

    try:
        df.to_excel(filename, index=False, engine='openpyxl') # Specify engine
        logger.info(f"Successfully exported {len(df)} chunks to '{filename}'")
    except Exception as e:
        logger.error(f"Failed to export to Excel: {e}")
        logger.error("Make sure you have 'openpyxl' installed: pip install openpyxl")

def main():
    if not QDRANT_API_KEY and QDRANT_HOST and "cloud.qdrant.co" in QDRANT_HOST: # Basic check for Qdrant Cloud
        logger.error("QDRANT_API_KEY is required for Qdrant Cloud but not found. Please set it in your .env file or environment variables.")
        return

    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        if QDRANT_API_KEY: # Qdrant Cloud or secured instance
             qdrant = QdrantClient(
                url=f"https://{QDRANT_HOST}:{QDRANT_PORT if QDRANT_PORT != 443 else ''}", # Qdrant Cloud typically uses 443 implicitly for https
                api_key=QDRANT_API_KEY
            )
        else: # Local, unsecured instance
            qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Test connection
        qdrant.get_collections()
        logger.info("Successfully connected to Qdrant.")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return

    # --- Parameters for fetching ---
    # Set to None to fetch ALL chunks, or a number like 10, 50, 100 to fetch only a few
    MAX_TO_FETCH = None # or e.g., 20
    SCROLL_BATCH_SIZE = 50 # How many to retrieve per API call

    fetched_points = fetch_all_chunks(
        qdrant_client=qdrant,
        collection_name=COLLECTION_NAME,
        limit_per_scroll=SCROLL_BATCH_SIZE,
        max_chunks_to_fetch=MAX_TO_FETCH
    )

    if fetched_points:
        export_chunks_to_excel(fetched_points, OUTPUT_EXCEL_FILE)
    else:
        logger.info(f"No chunks found or fetched from collection '{COLLECTION_NAME}'. Nothing to export.")


if __name__ == "__main__":
    main()