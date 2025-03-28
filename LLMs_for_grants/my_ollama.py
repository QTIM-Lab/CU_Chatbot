"""The agent that calls the API"""

from typing import List
import os

import pandas as pd

from ollama import chat
from ollama import ChatResponse

from langchain_community.document_loaders import PyPDFLoader

class OllamaModel:
    """The agent that calls the ollama API"""

    def __init__(self, model: str, prompt: str = None):
        """Build agent"""
        self.model = model
        self.prompt = prompt
        self.loaded_pdfs = False

    def initialize_system_prompt(self, rfa: str, sa: str):
        """Initialize the system prompt"""
        loader = PyPDFLoader(rfa) 
        rfa_text = "\n".join([doc.page_content for doc in loader.load()])

        loader = PyPDFLoader(sa) 
        sa_text = "\n".join([doc.page_content for doc in loader.load()])

        self.prompt = self.prompt.format(rfa=rfa_text, sa=sa_text)
        
    def __call__(self, text: str, history: List[str] = [], rfa: str = None, sa: str = None) -> str:
        """Send a chat message to the ollama API"""

        if not self.loaded_pdfs:
            self.initialize_system_prompt(rfa, sa)
            self.loaded_pdfs = True
        
        response: ChatResponse = chat(model=self.model, messages= [{'role': 'system', 'content': self.prompt}] + history + [
            {'role': 'user', 'content': text},
            ])
        
        return response.message.content
