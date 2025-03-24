import os
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

class OpenAI_Model:

    def __init__(self, deployment_id: str = 'got-4o'):
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
   
    def __call__(self, message, history):
        """Interaction with OpenAI model"""
        response = self.client.chat.completions.create(
            model=self.deployment_id,
            messages=[{"role": "user", "content": message}]
        )
        # print(response)
        return response.choices[0].message.content
