from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from utils.data_utils import ingest_repo
from utils.llm_utils import get_answer
from utils.injectors import initialize
from utils.custom_classes import Ingest, Generate

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_repo_from_url(ingest: Ingest):
    print(ingest)
    status_flag = ingest_repo(
        repo_url=ingest.repo_url,
        insert_custom_embeddings=ingest.insert_custom_embeddings,
    )
    if status_flag:
        return JSONResponse(
            content={"Info": "Ingested to Vector DB"},
            status_code=status.HTTP_201_CREATED,
        )
    return JSONResponse(
        content={"Error": "Failed to Ingest"}, status_code=status.HTTP_400_BAD_REQUEST
    )


@app.post("/generate")
async def generate_reponse(generate: Generate):
    response = get_answer(query=generate.query, repo_url=generate.repo_url)
    return JSONResponse(content=response, status_code=status.HTTP_200_OK)
