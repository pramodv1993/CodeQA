import toml

from qdrant_client import QdrantClient
from langchain.chat_models import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from utils.custom_classes import CodeT5PlusEmbeddings

qdrant_client = None
code_embeddings_model = None
llm = None
config = None
prompt_config = None


async def initialize():
    global qdrant_client, code_embeddings_model, config, llm, prompt_config
    code_embeddings_model = CodeT5PlusEmbeddings()
    config = toml.load("configs/properties.toml")
    qdrant_client = QdrantClient(
        url=config.get("VectorStore").get("url"), prefer_grpc=True
    )
    llm = ChatOpenAI(model_name=config.get("LLM").get("model_name"))
    prompt_config = toml.load("configs/prompts.toml")


def prompts() -> dict:
    return prompt_config


def llm_instance() -> BaseChatModel:
    return llm


def qdrant_client_instance() -> QdrantClient:
    return qdrant_client


def config_instance() -> dict:
    return config


def embeddings_model_instance() -> CodeT5PlusEmbeddings:
    return code_embeddings_model
