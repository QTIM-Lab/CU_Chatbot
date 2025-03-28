import os
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from document_handling import create_chroma_db_from_file
class OpenAI_Model:

    def __init__(self, deployment_id: str = 'got-4o', prompt: str = ''):
        default_credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            default_credential, "https://cognitiveservices.azure.com/.default"
        )

        load_dotenv()

        self.client = AzureOpenAI(
            # below this line, code is identical to https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart
            api_version="2024-02-01",
            azure_endpoint = os.getenv('OPENAI_API_BASE'),
            azure_ad_token_provider=token_provider
        )
        self.deployment_id = deployment_id
        self.loaded_pdfs = False
        self.prompt = prompt

    def initialize_rag(self, file_path: str):
        """Initialize the RAG database"""
        self.rag_db = create_chroma_db_from_file(file_path, self.model, os.path.basename(file_path).split('.')[0])

    def initialize_system_prompt(self, rfa: str, sa: str):
        """Initialize the system prompt"""
        loader = PyPDFLoader(rfa) 
        rfa_text = "\n".join([doc.page_content for doc in loader.load()])

        loader = PyPDFLoader(sa) 
        sa_text = "\n".join([doc.page_content for doc in loader.load()])

        self.prompt = self.prompt.format(rfa=rfa_text, sa=sa_text)
        # self.prompt = self.prompt.format(sa=sa_text)

    def __call__(self, message, history, rfa: str, sa: str):
        """Interaction with OpenAI model"""
        
        if not self.loaded_pdfs:
            self.initialize_system_prompt(rfa, sa)
            # self.initialize_rag(rfa)
            self.loaded_pdfs = True

        response = self.client.chat.completions.create(
            model=self.deployment_id,
            messages= [{'role': 'system', 'content': self.prompt}] + history + [{"role": "user", "content": message}]
        )
        # print(response)
        return response.choices[0].message.content
