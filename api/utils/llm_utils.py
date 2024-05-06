from typing import List

from langchain_community.vectorstores import Qdrant
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from utils.injectors import (
    embeddings_model_instance,
    qdrant_client_instance,
    config_instance,
    llm_instance,
    prompts,
)


def _add_source_info_to_result(result: str, doc_objs: List):
    sources = set(
        [
            doc_obj.metadata.get("file_level_metadata").get("file_name")
            for doc_obj in doc_objs
        ]
    )
    sources = "\n".join(sources)
    result += f"\n\nReferred Files: {sources}"
    return result


def get_answer(query: str, repo_url: str) -> str:
    repo_name = repo_url.split(".git")[0].split("/")[-1]
    config = config_instance()
    llm = llm_instance()
    embedding_model = embeddings_model_instance()
    qdrant = Qdrant(
        client=qdrant_client_instance(),
        collection_name=repo_name,
        vector_name=config.get("VectorStore").get("vector_col_name"),
        embeddings=embedding_model,
    )
    similar_doc_objects = qdrant.search(
        query=query,
        k=config.get("VectorStore").get("top_k"),
        search_type=config.get("VectorStore").get("search_type"),
    )

    messages = prompts().get("RAG").get("CHAT")
    prompt = ChatPromptTemplate.from_messages(
        [(role, content) for msg in messages for role, content in msg.items()]
    )
    chain = create_stuff_documents_chain(llm, prompt) | StrOutputParser()
    response = chain.invoke({"question": query, "context": similar_doc_objects})
    response = _add_source_info_to_result(response, similar_doc_objects)
    return response
