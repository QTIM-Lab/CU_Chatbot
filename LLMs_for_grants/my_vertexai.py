from multiprocessing import Pool
import time
from typing import Iterable, Optional

from tqdm import tqdm
import pandas as pd

import vertexai
from vertexai.generative_models import GenerativeModel

import google.auth
from google.auth.transport import requests

def get_credentials():
  credentials = None
  try:
    credentials, project_id = google.auth.default()
    # credentials = credentials.with_gdch_audience(f'https://{OCR_ENDPOINT}:443')
    req = requests.Request()
    credentials.refresh(req)
  except Exception as e:
    print("Caught exception" + str(e))
    raise e
  return credentials

class VertexAI():
    """Implements functionalities from VertexAI"""

    def __init__(self, model_name: str) -> None:

        vertexai.init(location="us-central1", credentials=get_credentials())

        self.model = GenerativeModel(model_name)

    def __call__(self, prompt: str, history: list) -> dict:
        """Process a single note and returns the answer"""

        return self.model.generate_content(prompt)
        