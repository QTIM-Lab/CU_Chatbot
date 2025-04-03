"""Generic chatbot model class"""

class ChatbotModel:
    def __init__(self):
        self.prompt_files = {'grant_rfa': 'prompts/grant_rfa_prompt.txt',
                            'grant': 'prompts/grant_prompt.txt',
                            'rfa': 'prompts/rfa_prompt.txt'}
    
    def read_prompt(self, prompt_name: str):
        """Read the prompt from the file"""
        with open(self.prompt_files[prompt_name], 'r') as f:
            prompt = f.read()
        return prompt

    def __call__(self, message, history):
        """Send a chat message to the model"""
        raise NotImplementedError("Subclasses must implement this method")

