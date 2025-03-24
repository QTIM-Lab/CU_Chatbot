"""
docker run --rm --gpus=all -it -v .:/usr/src/app --network=host llms_for_grants
-p 7860:7860 - not needed anymore
"""

from functools import partial
import gradio as gr

from my_vertexai import VertexAI
from my_openai import OpenAI_Model

def chat_with_model(message, history, model):
    """Chat with the model"""
    return model(message, history)

def initialize_model(model_name: str):
    """Returns chat function to use to use"""

    if model_name == 'gemini-1.5-flash-001':
        model = VertexAI(model_name)
    elif model_name == 'got-4o':
        model = OpenAI_Model(model_name)
    elif model_name == 'llama3.2':
        return gr.load_chat("http://localhost:11434/v1/", model=model_name, token='***')
    else:
        raise NotImplementedError(f'Model {model_name} not supported')
    return partial(chat_with_model, model=model)

if __name__ == '__main__':
    chat_fn = initialize_model('got-4o')
    iface = gr.ChatInterface(fn=chat_fn, type="messages").launch()
