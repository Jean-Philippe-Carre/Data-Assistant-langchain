import os
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# "openai_key" is the variable in which i declared the key in .env
API_KEY = os.getenv("openai_key")
