from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv
load_dotenv()


def get_mistralai_model():
    """
    Get a MistralAI model by name or path.

    Args:
        model_name (str): The name of the MistralAI model.
        model_path (str, optional): The path to the MistralAI model. Defaults to None.

    Returns:
        MistralAIModel: The MistralAI model instance.
    """
    return ChatMistralAI(api_key=os.environ.get("MISTRALAI_KEY"), model="mistral-large-latest", temperature=0)
