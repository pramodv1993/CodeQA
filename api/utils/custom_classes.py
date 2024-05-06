from typing import List, Any

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
        model_path: str = "models/codet5p-110m-embedding/snapshots/94f88f95672b1d4b0cc715c6011001a74f892bdd",
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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer.batch_encode_plus(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        return self.model(inputs["input_ids"]).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model(
            self.tokenizer.encode(query, return_tensors="pt").to(self.device)
        ).tolist()[0]
