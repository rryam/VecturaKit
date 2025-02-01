#!/usr/bin/env python3
import os
import subprocess
from together import Together

# Initialize the Together client using your API key.
# Ensure that TOGETHER_API_KEY is set as an environment variable.
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# The system prompt instructs the model on what to do.
SYSTEM_PROMPT = (
    "You are a helpful assistant that reads the entire code base and rewrites the README.md file "
    "to provide clear instructions, describe the package, list dependencies, and usage examples. "
    "Please analyze the code and produce an updated README."
)

def get_codebase_summary():
    """
    Generates a summary of the codebase by listing tracked files and their content.
    You might extend this function to include more context if needed.
    """
    files = subprocess.check_output(["git", "ls-files"]).decode("utf-8").splitlines()
    summary = ""
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"Error reading file: {e}"
        summary += f"--- {file} ---\n{content}\n\n"
    return summary

def call_togetherai(prompt):
    """
    Uses the Together AI Python SDK to generate an updated README.
    It sends two messages: one with the system instructions and another with the codebase context.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    # Adjust the model name as needed; here we're using an instruct-turbo model.
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages
    )
    return response.choices[0].message.content

def main():
    # Gather codebase context to inform the README generation.
    codebase_context = get_codebase_summary()
    # Build the user prompt with the context from the codebase.
    full_prompt = f"Codebase files:\n{codebase_context}"
    
    updated_readme = call_togetherai(full_prompt)
    if updated_readme:
        with open("README.md", "w") as f:
            f.write(updated_readme)
        print("README.md has been updated.")
    else:
        print("Failed to update README.md.")

if __name__ == "__main__":
    main()