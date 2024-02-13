from openai import OpenAI

client = OpenAI()
import os

api_key = os.getenv("OPENAI_API_KEY")

with open("conversations.jsonl", "rb") as data:
    response = client.files.create(file=data,
    purpose="fine-tune")

file_id = getattr(response, 'id', None)
if file_id:
    print(f"File uploaded successfully with ID: {file_id}")
else:
    print("File upload failed.")