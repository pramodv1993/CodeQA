from typing import List
import uuid

from langchain_community.vectorstores import Qdrant
from langchain_core.documents.base import Document
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

from utils.custom_classes import RepoFile
from utils.injectors import (
    embeddings_model_instance,
    qdrant_client_instance,
    config_instance,
)


def _batch_insert(
    chunked_docs: List[Document],
    custom_chunk_embeddings: List,
    repo_name: str,
    batch_size: int = 500,
):
    points = []
    qdrant_client = qdrant_client_instance()
    config = config_instance()
    # wrap chunks as PointStruct objects
    for chunk_doc, custom_chunk_embedding in zip(chunked_docs, custom_chunk_embeddings):
        points.append(
            PointStruct(
                id=uuid.uuid4().hex,
                payload={
                    "metadata": chunk_doc.metadata,
                    "page_content": chunk_doc.page_content,
                },
                vector={
                    config.get("VectorStore").get(
                        "vector_col_name"
                    ): custom_chunk_embedding
                },
            )
        )
    # create collection if not exist
    if not qdrant_client.collection_exists(repo_name):
        print(f"Collection does not exist, hence creating collection : {repo_name}")
        qdrant_client.recreate_collection(
            collection_name=repo_name,
            vectors_config={
                config.get("VectorStore").get("vector_col_name"): VectorParams(
                    size=256, distance=Distance.COSINE
                )
            },
        )
    # insert in batches
    for i in range(0, len(points), batch_size):
        print(f"Inserting {i}-{i+batch_size} points")
        qdrant_client.upsert(
            collection_name=repo_name, points=points[i : i + batch_size]
        )


def _update_custom_embeddings(chunked_docs: List[Document]) -> List[List]:
    """
    An attempt to incorporate Contextual RAG
    eg: fusing the metadata and contextual info of the chunk into the embedding
    """
    custom_chunk_embeddings = []
    print("Computing custom embeddings for chunks..")
    for chunk in chunked_docs:
        chunk.page_content
        chunk.metadata
        # @TODO add fusing here-- eg: embed(metadata) + embed(content) + embed(summary) etc
        embedding = [1] * 256
        custom_chunk_embeddings.append(embedding)
    return custom_chunk_embeddings


def embed_and_store(
    repo_files: List[RepoFile],
    insert_custom_embeddings: bool = False,
) -> bool:
    # Using repo name as the collection name
    chunked_docs = []
    repo_name = repo_files[0].get("repo_name")
    config = config_instance()
    for repo_file in repo_files:
        chunked_docs.extend(repo_file.get("chunks"))
    try:
        if insert_custom_embeddings:
            print("Using custom embeddings..")
            custom_chunk_embeddings = _update_custom_embeddings(chunked_docs)
            _batch_insert(
                chunked_docs=chunked_docs,
                custom_chunk_embeddings=custom_chunk_embeddings,
                repo_name=repo_name,
            )
        else:
            print("Using langchain to create embeddings..")
            Qdrant.from_documents(
                documents=chunked_docs,
                embedding=embeddings_model_instance(),
                vector_name=config.get("VectorStore").get("vector_col_name"),
                url=config_instance().get("VectorStore").get("url"),
                collection_name=repo_name,
                prefer_grpc=True,
            )
    except Exception as e:
        print(e)
        return False
    return True
