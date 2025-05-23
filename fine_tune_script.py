import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in .env file.")
    exit()

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Configuration ---
TRAIN_FILE_PATH = "train_data.jsonl" # Path to your training data
VALIDATION_FILE_PATH = "validation_data.jsonl" # Path to your validation data (optional)
MODEL_TO_FINETUNE = "gpt-3.5-turbo-0125" # Or another compatible gpt-3.5-turbo model
CUSTOM_SUFFIX = "gurudev-bio-extractor-v1" # A descriptive suffix for your fine-tuned model's name

def upload_file_to_openai(filepath, purpose="fine-tune"):
    """Uploads a file to OpenAI and returns the file ID."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, "rb") as f:
            response = client.files.create(file=f, purpose=purpose)
        print(f"File '{filepath}' uploaded successfully. File ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error uploading file '{filepath}': {e}")
        return None

def create_fine_tuning_job(training_file_id, validation_file_id=None, model=MODEL_TO_FINETUNE, suffix=CUSTOM_SUFFIX):
    """Creates a fine-tuning job."""
    try:
        job_params = {
            "training_file": training_file_id,
            "model": model,
            "suffix": suffix,
            # You can add hyperparameters here if needed, e.g., "hyperparameters": {"n_epochs": 3}
        }
        if validation_file_id:
            job_params["validation_file"] = validation_file_id

        response = client.fine_tuning.jobs.create(**job_params)
        print(f"Fine-tuning job created successfully. Job ID: {response.id}, Status: {response.status}")
        return response.id, response.status
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None, None

def monitor_fine_tuning_job(job_id):
    """Monitors the fine-tuning job until completion or failure."""
    print(f"Monitoring job {job_id}...")
    while True:
        try:
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            status = job_status.status
            fine_tuned_model_id = job_status.fine_tuned_model

            print(f"Job Status: {status}. Fine-tuned model ID: {fine_tuned_model_id if fine_tuned_model else 'Not available yet.'}")

            if status == "succeeded":
                print(f"Fine-tuning job {job_id} succeeded!")
                print(f"Fine-tuned Model ID: {fine_tuned_model_id}")
                # You can also retrieve and print events:
                # events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=50)
                # for event in reversed(events.data):
                # print(f" - {event.created_at}: {event.message}")
                return fine_tuned_model_id
            elif status in ["failed", "cancelled"]:
                print(f"Fine-tuning job {job_id} {status}. Error: {job_status.error}")
                return None
            
            # List events for more details during processing
            events_response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
            for event in reversed(events_response.data): # Show newest events first for this poll
                print(f"  Event ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.created_at))}): {event.message}")


        except Exception as e:
            print(f"Error retrieving job status for {job_id}: {e}")
            # Decide if you want to break or continue retrying on API errors
        
        time.sleep(60) # Wait for 60 seconds before checking again

if __name__ == "__main__":
    print("Starting fine-tuning process...")

    # 1. Upload training data
    print("\n--- Step 1: Uploading Training Data ---")
    training_file_id = upload_file_to_openai(TRAIN_FILE_PATH)
    if not training_file_id:
        print("Failed to upload training file. Exiting.")
        exit()

    # 2. Upload validation data (optional)
    validation_file_id = None
    if os.path.exists(VALIDATION_FILE_PATH):
        print("\n--- Step 2: Uploading Validation Data ---")
        validation_file_id = upload_file_to_openai(VALIDATION_FILE_PATH)
        if not validation_file_id:
            print("Failed to upload validation file, but will proceed without it.")
    else:
        print(f"\n--- Step 2: Validation file '{VALIDATION_FILE_PATH}' not found. Proceeding without validation data. ---")


    # 3. Create fine-tuning job
    print("\n--- Step 3: Creating Fine-Tuning Job ---")
    job_id, initial_status = create_fine_tuning_job(training_file_id, validation_file_id)
    if not job_id:
        print("Failed to create fine-tuning job. Exiting.")
        exit()

    # 4. Monitor job (optional, but highly recommended)
    if initial_status not in ["failed", "cancelled", "succeeded"]:
        print("\n--- Step 4: Monitoring Fine-Tuning Job ---")
        fine_tuned_model_name = monitor_fine_tuning_job(job_id)
        if fine_tuned_model_name:
            print(f"\nProcess complete! Your fine-tuned model is: {fine_tuned_model_name}")
            print("You can now use this model ID in your API calls.")
        else:
            print("\nFine-tuning process did not complete successfully.")
    else:
        print(f"\nJob {job_id} already completed or failed with status: {initial_status}.")
        if initial_status == "succeeded":
            job_info = client.fine_tuning.jobs.retrieve(job_id)
            print(f"Your fine-tuned model ID is: {job_info.fine_tuned_model}")