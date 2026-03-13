import os

from google import genai
from google.genai.types import HttpOptions

from data_generation.task_level.runtime.client import (
    DEFAULT_LOCATION,
    load_dotenv_file,
)


load_dotenv_file()

client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION),
    http_options=HttpOptions(api_version="v1"),
)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="How does AI work?",
)
print(response.text)
