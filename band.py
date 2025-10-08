import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
models = openai.Model.list()
print([m.id for m in models.data])
