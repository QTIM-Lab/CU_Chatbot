"""The agent that calls the API"""

from typing import List
import os

import pandas as pd

from ollama import chat
from ollama import ChatResponse
from langchain_community.vectorstores import Chroma

from document_handling import get_context_for_query, create_chroma_db_from_file

class OllamaModel:
    """The agent that calls the ollama API"""

    def __init__(self, model: str, prompt: str = None):
        """Build agent"""
        self.model = model
        self.prompt = prompt
        self.rag_db = None

    def initialize_rag(self, sa: str):
        """Initialize the RAG database"""
        self.prompt = self.prompt.format(context=get_context_for_query(sa, self.rag_db))
        # self.rag_db = create_chroma_db_from_file(sa, self.model, os.path.basename(sa).split('.')[0])
    
    def __call__(self, text: str, history: List[str] = [], sa: str = None) -> str:
        """Send a chat message to the ollama API"""

        if self.rag_db is None and sa is not None:
            self.initialize_rag(sa.name)

        if not self.rag_db is None:
            # get context for query
            context = get_context_for_query(text, self.rag_db)
            if not self.prompt is None:
                # add context to prompt
                text = self.prompt.format(context=context, query=text)
        
        response: ChatResponse = chat(model=self.model, messages= [self.prompt] + history + [
            {'role': 'user', 'content': text},
            ])
        
        return response.message.content
