#!/usr/bin/env python3
import os
import subprocess
from google import genai

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

def call_geminiai(prompt):
    """
    Uses the Google GenAI Python SDK with Gemini to generate an updated README.
    It generates content based on the provided prompt.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(model='gemini-2.0-flash-exp', contents=prompt)
    return response.text

def main():
    # Gather codebase context to inform the README generation.
    codebase_context = get_codebase_summary()
    # Build the user prompt with the context from the codebase.
    full_prompt = f"{SYSTEM_PROMPT}\n\nCodebase files:\n{codebase_context}"
    
    updated_readme = call_geminiai(full_prompt)
    if updated_readme:
        with open("README.md", "w") as f:
            f.write(updated_readme)
        print("README.md has been updated.")
    else:
        print("Failed to update README.md.")

if __name__ == "__main__":
    main()