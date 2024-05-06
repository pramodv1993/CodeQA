## CodeQA
Problem Statement: <br>
Given a URL of a Github repository, the proposed solution enables the user to have a priliminary understanding of the repository by asking the system in a conversation style setup.

### High Level Design:
<img src="images/design_codeqa.jpg" alt="drawing" width="500"/><br>
3 main modules, each of which function as a standlone dockerized microservice:
- API: That captures the main tasks of downloading the repository, processing it, embedding the same and so on. More details can be found [here](/api/README.md)
- UI: Simple interface to see the solution in action.
- VecDB: A vector database that supports CRUD of vector embeddings as well as some metadata information.

### Demo:
[Insert Video here]

### QuickStart:
- Include a `.env` file with a key for `OPENAI_API_KEY=""` in the `API` module (ie in [this](/api/) path)
- Download the embedding model from [here](https://drive.google.com/drive/folders/1LjC2qsG69-PWuv8No8l4vtVGi11bTfRN?usp=sharing) and place in in [this](/api/models/) location
- The microservices are encapsulated as composable docker services. Hence run <br> `docker-compose up ---build` at the [root](/) location.
- You can find each of the modules in the following URLs:
    - UI: [localhost:8000](localhost:8000)
    - API: [localhost:8001/docs](localhost:8001/docs)
    - Qdrant Dashboard: [localhost:6333/dashboard](localhost:6333/dashboard)

### More technical details:
- UI:
    - [Streamlit](https://streamlit.io/) is used for building a basic interactive app.
    - A screenshot:
    <img src="images/ui.png" alt="drawing" width="400"/><br>
- API:
    - [Fast API](https://fastapi.tiangolo.com/) is used for implementing RESTful services. More details about the supported APIs can be found [here](/api/).
- VectorDB
    - Self-hosted [Qdrant](https://qdrant.tech/) database is used as a vector database.

### Next steps:
- Improvements can be done at several places, Some of them (but not limited to) could be:
    - Filtering strategies while downloading the repo
    - Detecting programming language of the scripts and performing appropriate cleaning strategies
    - Creating richer metadata for the scripts at both document and chunk level such as summaries of functions, comments, function_names etc.
    - More advanced strategies while embedding the scripts, to bring about "Contextual RAG".
- I have tried to add comments (`@TODO`) in appropriate places in the scripts.


