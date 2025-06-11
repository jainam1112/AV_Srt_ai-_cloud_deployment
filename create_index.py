from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get credentials
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "AV_srt_recognization"

def create_index():
    try:
        # Initialize client
        qdrant_client = QdrantClient(
            url=f"https://{QDRANT_HOST}:6333",
            api_key=QDRANT_API_KEY
        )
        
        # Create index for transcript_name field
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="transcript_name",
            field_schema=models.PayloadFieldSchema.KEYWORD
        )
        
        logger.info(f"Successfully created index for 'transcript_name' in collection '{COLLECTION_NAME}'")
        
    except Exception as e:
        logger.error(f"Failed to create index: {str(e)}")

if __name__ == "__main__":
    create_index()