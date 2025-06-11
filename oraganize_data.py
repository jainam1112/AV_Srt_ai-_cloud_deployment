import json
import os

def organize_training_data(input_filepath, output_filepath):
    """
    Reads a JSON Lines file, validates the expected chat format,
    and writes the validated data to a new JSON Lines file.

    Each line in the input file should be a JSON object like:
    {"messages": [{"role": "user", "content": "user_prompt"}, {"role": "assistant", "content": "assistant_completion"}]}
    The assistant's content is expected to be a string, which itself might be a JSON string.
    """
    organized_entries = []
    line_number = 0
    error_count = 0

    print(f"Starting to organize file: {input_filepath}")

    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found.")
        return

    with open(input_filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            line_number += 1
            stripped_line = line.strip()
            if not stripped_line:  # Skip empty lines
                continue

            try:
                data_entry = json.loads(stripped_line)
            except json.JSONDecodeError as e:
                print(f"Line {line_number}: Invalid JSON - {e}. Skipping this line.")
                error_count += 1
                continue

            valid_entry = True
            if not isinstance(data_entry, dict):
                print(f"Line {line_number}: Entry is not a JSON object. Skipping this line.")
                error_count += 1
                valid_entry = False
            elif "messages" not in data_entry:
                print(f"Line {line_number}: Missing 'messages' key. Skipping this line.")
                error_count += 1
                valid_entry = False
            else:
                messages = data_entry["messages"]
                if not isinstance(messages, list) or len(messages) != 2:
                    print(f"Line {line_number}: 'messages' should be a list of 2 items. Skipping this line.")
                    error_count += 1
                    valid_entry = False
                else:
                    # Validate user message
                    user_msg = messages[0]
                    if not (isinstance(user_msg, dict) and
                            user_msg.get("role") == "user" and
                            "content" in user_msg and
                            isinstance(user_msg["content"], str)):
                        print(f"Line {line_number}: Invalid user message format. Skipping this line.")
                        error_count += 1
                        valid_entry = False

                    # Validate assistant message
                    assistant_msg = messages[1]
                    if not (isinstance(assistant_msg, dict) and
                            assistant_msg.get("role") == "assistant" and
                            "content" in assistant_msg and
                            isinstance(assistant_msg["content"], str)):
                        print(f"Line {line_number}: Invalid assistant message format. Skipping this line.")
                        error_count += 1
                        valid_entry = False
                    
                    # Optional: Validate if assistant's content is a valid JSON string
                    # This check is useful if the assistant is always expected to produce valid JSON.
                    if valid_entry: # Only check if basic structure is fine
                        try:
                            json.loads(assistant_msg["content"])
                        except json.JSONDecodeError:
                            # This might be acceptable depending on the fine-tuning task.
                            # If the assistant sometimes produces non-JSON strings, this is not an error for the script.
                            # print(f"Line {line_number}: Warning - Assistant content is not a valid JSON string.")
                            pass


            if valid_entry:
                organized_entries.append(data_entry)

    if not organized_entries:
        if error_count > 0:
            print(f"\nNo valid entries found after processing. {error_count} lines had issues.")
        else:
            print(f"\nNo entries found in {input_filepath}.")
        return

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for entry in organized_entries:
            outfile.write(json.dumps(entry) + '\n')

    print(f"\nSuccessfully organized {len(organized_entries)} entries.")
    print(f"Output written to: {output_filepath}")
    if error_count > 0:
        print(f"{error_count} lines had issues and were skipped (see details above).")

if __name__ == "__main__":
    # Assuming train.json is in the same directory as the script,
    # or in the current working directory when the script is run.
    # The user's workspace is c:\Users\jaina\satsang_search_app\cloud_deployment
    input_filename = "train.json"
    output_filename = "organized_train_data.jsonl" # .jsonl is a common extension for JSON Lines

    # Construct paths relative to the script's location or current working directory
    # For VS Code, if the workspace is set correctly, these relative paths should work.
    
    # Get the directory of the current script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # input_file_path = os.path.join(script_dir, input_filename)
    # output_file_path = os.path.join(script_dir, output_filename)

    # Simpler approach if running from the workspace root:
    input_file_path = input_filename
    output_file_path = output_filename


    organize_training_data(input_file_path, output_file_path)