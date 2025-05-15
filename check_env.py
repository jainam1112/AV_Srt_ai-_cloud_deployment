import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print environment variables (masking the API key)
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print(f"QDRANT_HOST: {QDRANT_HOST}")
print(f"QDRANT_API_KEY: {'*' * len(QDRANT_API_KEY) if QDRANT_API_KEY else 'Not set'}")