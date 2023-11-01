import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.tokenizer_paths=["/shared_data0/arobey1/llama-2-7b-chat-hf"]
    config.model_paths=["/shared_data0/arobey1/llama-2-7b-chat-hf"]
    config.conversation_templates=['llama-2']

    return config