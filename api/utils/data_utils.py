import os
from typing import List
import re

from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from utils.custom_classes import RepoFile
from utils.vector_utils import embed_and_store, _is_collection_exists


def _is_valid_url(repo_url: str):
    return re.match(
        r"https?:\/\/(?:www\.)?github\.com\/[a-zA-Z0-9-]+\/[a-zA-Z0-9-]+(?:\/)?(?:\w+)?(?:\/)?(?:\w+)?",
        repo_url,
    )


def _download_repo(url: str):
    repo = None
    repo_dir = url.split("/")[-1].split(".")[0]
    try:
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)
            print("Downloading Repo..")
            repo = Repo.clone_from(url, repo_dir)
        else:
            print("Repo already downloaded..")
            repo = Repo.init(repo_dir)
    except Exception as e:
        print(e)
    return repo


def _filter_text_files(repo: Repo) -> List[RepoFile]:
    # @TODO add additional processing steps for each file
    print("Filtering files that has text..")
    is_hidden_dir = lambda path: path.startswith(".")
    get_file_name = lambda path: path.split("/")[-1]
    is_notebook = lambda path: path.endswith(".ipynb")
    repo_name = repo.remotes.origin.url.split(".git")[0].split("/")[-1]
    files = []
    names = []
    for blob in repo.tree().traverse():
        if (
            not is_hidden_dir(blob.path)
            and not is_notebook(blob.path)
            and blob.type == "blob"
        ):
            try:
                data = blob.data_stream.read().decode("utf-8")
                if len(data) == 0:
                    continue
                print(f"Adding contents of the file {blob.path}")
                files.append(
                    RepoFile(
                        content=data,
                        repo_name=repo_name,
                        # metadata can be at the doc level or the chunk level
                        file_level_metadata={
                            "file_name": get_file_name(blob.path),
                            "path": blob.path,
                        },
                    )
                )
                names.append(get_file_name(blob.path))
            except Exception:
                print(f"Could not parse {blob.path}")
    print(f"Files parsed: {names}")
    return files


def _clean_scripts(repo_files: List[Repo]):
    # @TODO add additional cleaning steps
    print("Cleaning scripts..")
    for repo_file in repo_files:
        # remove trailing spaces and new lines
        repo_file.set("content", repo_file.get("content").rstrip())
    return repo_files


def _update_file_level_metadata(repo_files: List[RepoFile]) -> List[RepoFile]:
    """@TODO add additional file_level_metadata such as functions names in script, comments etc"""
    print("Updating File level metadata..")
    for repo_file in repo_files:
        # length of whole script
        content = repo_file.get("content")
        repo_file.get("file_level_metadata", {}).update({"length": len(content)})
    return repo_files


def _chunk_repo_files(
    repo_files: List[RepoFile],
    prog_language: str = "PYTHON",
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> List[RepoFile]:
    print("Chunking files..")
    splitter_for_lang = RecursiveCharacterTextSplitter.from_language(
        language=Language._member_map_[prog_language],
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
    )
    for file in repo_files:
        # @TODO detect the programing language type based on the file content to get better chunking
        content = file.get("content")
        file_metadata = {"file_level_metadata": file.get("file_level_metadata")}
        chunked_docs = splitter_for_lang.create_documents(
            [content], metadatas=[file_metadata]
        )
        file.set("chunks", chunked_docs)
    return repo_files


def _update_chunk_level_metadata(repo_files: List[RepoFile]) -> List[RepoFile]:
    """#@TODO add additional chunk_level_metadata if needed"""
    print("Updating chunk level metadata..")
    for repo_file in repo_files:
        chunks = repo_file.get("chunks", [])
        for chunk_no, chunk_doc in enumerate(chunks):
            chunk_doc.metadata.update(
                {
                    "chunk_level_metadata": {
                        "chunk_no": chunk_no,
                        "length": len(chunk_doc.page_content),
                    }
                }
            )
    return repo_files


def _store_chunks(repo_files: List[RepoFile], insert_custom_embeddings: bool):
    print("Embedding and storing in Qdrant..")
    return embed_and_store(
        repo_files, insert_custom_embeddings=insert_custom_embeddings
    )


def _preprocess_data(repo: Repo) -> List[RepoFile]:
    repo_files = _filter_text_files(repo)
    repo_files = _clean_scripts(repo_files=repo_files)
    return repo_files


def ingest_repo(
    repo_url: str, prog_language: str = "PYTHON", insert_custom_embeddings: bool = False
) -> bool:
    repo_dir = repo_url.split("/")[-1].split(".")[0]
    if not _is_valid_url(repo_url):
        raise Exception("Invalid repo URL!")
    elif _is_collection_exists(repo_dir):
        return True
    try:
        repo: Repo = _download_repo(repo_url)
        repo_files: List[RepoFile] = _preprocess_data(repo)
        repo_files: List[RepoFile] = _update_file_level_metadata(repo_files)
        repo_files: List[RepoFile] = _chunk_repo_files(repo_files, prog_language)
        repo_files: List[RepoFile] = _update_chunk_level_metadata(repo_files)
        status_flag = _store_chunks(repo_files, insert_custom_embeddings)
        print(f"Status {status_flag}")
    except Exception as e:
        print(e)
        return False
    return status_flag
