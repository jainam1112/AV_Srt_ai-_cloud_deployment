import os
from openai import OpenAI

# --- Configuration ---
# IMPORTANT: Set your OpenAI API key as an environment variable for security.
# How to set environment variable:
# PowerShell: $env:OPENAI_API_KEY="your-api-key-here"
# CMD: set OPENAI_API_KEY=your-api-key-here
# Or, less securely, you can assign it directly:
# client = OpenAI(api_key="sk-your-api-key-here")
client = OpenAI() # Initializes with OPENAI_API_KEY environment variable

TRAINING_FILE_NAME = "organized_train_data.jsonl"
# Choose a base model. Check OpenAI documentation for the latest fine-tunable models.
# Examples: "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"
# GPT-4 models might have different availability for fine-tuning.
BASE_MODEL_ID = "gpt-3.5-turbo-0125"
# Optional: A suffix for your fine-tuned model's name
CUSTOM_MODEL_SUFFIX = "satsang_search_v1"

def upload_file_to_openai(filepath):
    """Uploads a file to OpenAI for fine-tuning."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        print(f"Uploading {filepath} to OpenAI...")
        with open(filepath, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="fine-tune")
        print(f"File uploaded successfully. File ID: {uploaded_file.id}")
        return uploaded_file.id
    except Exception as e:
        print(f"An error occurred during file upload: {e}")
        return None

def start_fine_tuning_job(training_file_id, base_model, suffix=None):
    """Starts a fine-tuning job on OpenAI."""
    if not training_file_id:
        print("Error: Training file ID is required to start a fine-tuning job.")
        return None
    try:
        print(f"Starting fine-tuning job with file ID: {training_file_id} on base model: {base_model}")
        job_params = {
            "training_file": training_file_id,
            "model": base_model
        }
        if suffix:
            job_params["suffix"] = suffix
        
        # You can also specify hyperparameters if needed, e.g.:
        # job_params["hyperparameters"] = {"n_epochs": 3}

        fine_tuning_job = client.fine_tuning.jobs.create(**job_params)
        print(f"Fine-tuning job started successfully. Job ID: {fine_tuning_job.id}")
        print("Monitor the job status on the OpenAI platform or via API.")
        print(f"Once completed, your model name will be similar to: {base_model}:{(suffix +':') if suffix else ''}<job-specific-id>")
        return fine_tuning_job
    except Exception as e:
        print(f"An error occurred while starting the fine-tuning job: {e}")
        return None

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
    else:
        # Construct the full path to the training file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        training_file_path = os.path.join(script_dir, TRAINING_FILE_NAME)

        # Step 1: Upload the training data
        uploaded_file_id = upload_file_to_openai(training_file_path)

        if uploaded_file_id:
            # Step 2: Start the fine-tuning job
            # It might be good to wait a few seconds for the file to be fully processed by OpenAI,
            # though often it's not strictly necessary.
            # import time
            # print("Waiting a few seconds for file processing...")
            # time.sleep(10) 
            
            job_details = start_fine_tuning_job(uploaded_file_id, BASE_MODEL_ID, CUSTOM_MODEL_SUFFIX)
            if job_details:
                print("\n--- Important Next Steps ---")
                print(f"1. Your fine-tuning Job ID is: {job_details.id}")
                print(f"2. Monitor progress: https://platform.openai.com/finetune (or use client.fine_tuning.jobs.retrieve('{job_details.id}'))")
                print("3. Once the job status is 'succeeded', your fine-tuned model will be available.")
                print("   You can list your fine-tuned models using 'client.models.list()' or find the name in the job details.")