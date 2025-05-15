from qdrant_client import QdrantClient
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

def test_connection():
    try:
        # Initialize client with direct URL
        qdrant_client = QdrantClient(
            url=f"https://{QDRANT_HOST}:6333",
            api_key=QDRANT_API_KEY
        )
        
        # Test connection by getting collections
        collections = qdrant_client.get_collections()
        logger.info(f"Successfully connected to Qdrant!")
        logger.info(f"Available collections: {collections}")
        return True
        
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Print connection details (masking API key)
    print(f"\nTesting connection to: https://{QDRANT_HOST}:6333")
    print(f"API Key: {'*' * len(QDRANT_API_KEY) if QDRANT_API_KEY else 'Not set'}")
    
    # Run test
    success = test_connection()
    print(f"\nConnection test: {'✅ Success' if success else '❌ Failed'}")