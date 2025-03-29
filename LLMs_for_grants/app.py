"""
docker run --rm --gpus=all -it -v .:/usr/src/app --network=host llms_for_grants
-p 7860:7860 - not needed anymore
"""

import os
from functools import partial
import gradio as gr

from my_vertexai import VertexAI
from my_openai import OpenAI_Model
from my_ollama import OllamaModel
from document_handling import create_chroma_db_from_file
from auth import check_auth
def chat_with_model(message, history, model):
    """Chat with the model"""
    return model(message, history)

def initialize_model(model_name: str):
    """Returns chat function to use to use"""
    
    # read prompt
    with open('prompts/sa_rfa_prompt.txt', 'r') as f:
        prompt = f.read()

    if model_name == 'gemini-1.5-flash-001':
        model = VertexAI(model_name)
    elif model_name == 'gpt-4o':
        model = OpenAI_Model(model_name, prompt)
    elif model_name == 'llama3.2':
        # return gr.load_chat("http://localhost:11434/v1/", model=model_name, token='***')
        return OllamaModel(model_name, prompt)
    else:
        raise NotImplementedError(f'Model {model_name} not supported')
    return model

if __name__ == '__main__':
    model = 'gpt-4o'

    # initialize chat function
    chat_fn = initialize_model(model)

    # launch gradio app
    with gr.Blocks() as demo:
        gr.Markdown("## NIH Grant Proposal Chatbot")
        rfa = gr.File(label="Upload RFA", file_types=[".pdf"])
        sa = gr.File(label="Upload Specific Aims", file_types=[".pdf"])
        rfa.upload(fn=chat_fn.reset_prompt)
        sa.upload(fn=chat_fn.reset_prompt)
        # chatbot = gr.Chatbot(type="messages")
        gr.ChatInterface(fn=chat_fn, type="messages",additional_inputs=[rfa, sa])
    
demo.launch(auth=check_auth)
# for CCTSI machine
# demo.launch(server_port=443, auth=check_auth, server_name='pccweb1016.ucdenver.pvt', ssl_verify=False, ssl_keyfile='key.pem', ssl_certfile='cert.pem')
