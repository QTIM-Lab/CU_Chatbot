import os
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from document_handling import create_chroma_db_from_file
from my_chatbot_model import ChatbotModel

class OpenAI_Model(ChatbotModel):

    def __init__(self, deployment_id: str = 'got-4o'):
        super().__init__()
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
        self.prompt = None

    def initialize_rag(self, file_path: str):
        """Initialize the RAG database"""
        self.rag_db = create_chroma_db_from_file(file_path, self.model, os.path.basename(file_path).split('.')[0])

    def initialize_system_prompt(self, rfa: str, sa: str):
        """Initialize the system prompt"""

        args = {'rfa': None, 'sa': None}
        if rfa is not None:
            loader = PyPDFLoader(rfa) 
            args['rfa'] = "\n".join([doc.page_content for doc in loader.load()])

        if sa is not None:
            loader = PyPDFLoader(sa) 
            args['sa'] = "\n".join([doc.page_content for doc in loader.load()])

        if 'rfa' in args and 'sa' in args:
            system_prompt = self.read_prompt('grant_rfa')
        elif 'rfa' in args:
            system_prompt = self.read_prompt('rfa')
        elif 'sa' in args:
            system_prompt = self.read_prompt('grant')
        else:
            raise ValueError('No valid documents provided')

        self.prompt = system_prompt.format(**args)
    
    def reset_prompt(self):
        """Reset the system prompt"""
        self.loaded_pdfs = False

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
