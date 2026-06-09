from typing import List, Any

import torch
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


# api request objects
class Ingest(BaseModel):
    repo_url: str = "https://github.com/zpqrtbnk/test-repo.git"
    insert_custom_embeddings: bool = False


class Generate(BaseModel):
    query: str = "What is the repo about?"
    repo_url: str = "https://github.com/zpqrtbnk/test-repo.git"


# Used for data processing
class RepoFile:
    """To manage whole file content, whole file and chunk level metadata"""

    def __init__(self, content: str = None, **kwargs):
        self.file = {}
        if content:
            self.file["content"] = content
        else:
            raise Exception("Content not provided to the file")
        self.file.update(kwargs)

    def get(self, key: str, default: Any = None):
        return self.file.get(key, default)

    def set(self, key: str, value: Any):
        self.file[key] = value

    def __str__(self) -> str:
        res = "----\n"
        for k, v in self.file.items():
            res += f"{k} : {v}\n"
        return res


# Custom embeddings class to be used in the LangChain pipeline
"""
CodeT5+
-span denoising - MLM for code
-unimodal (just code) followed by (text-code matching style training)
"""


class CodeT5PlusEmbeddings(Embeddings):
    # model: "Salesforce/codet5p-110m-embedding"
    def __init__(
        self,
        model_path: str = "models/codet5p-110m-embedding/snapshots/d9f3a534af4252f04b10cd9f78e05037542d10f6",
    ) -> None:
        # @TODO accept config from properties file
        # model_name = "Salesforce/codet5p-110m-embedding"
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(
            self.device
        )
        super().__init__()

    def embed_documents(
        self, texts: List[str], batch_size: int = 16
    ) -> List[List[float]]:
        # Embed in small batches: feeding a whole repo's chunks in a single
        # forward pass pads every chunk to the longest sequence and spikes
        # memory enough to OOM-kill the API process mid-request.
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer.batch_encode_plus(
                batch, return_tensors="pt", truncation=True, padding=True
            ).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(
                    inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
            embeddings.extend(batch_embeddings.tolist())
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        inputs = self.tokenizer.encode(
            query, return_tensors="pt", truncation=True
        ).to(self.device)
        with torch.no_grad():
            embedding = self.model(inputs)
        return embedding.tolist()[0]
