## APIs
- Supports 2 APIs:
    - Ingest (`/ingest`):
        - Given a URL, this API attemps to perform the following steps:
        <img src="../media/data_pipe.png" alt="drawing" width="800" /><br>
        - The API primarily leverages the [LangChain](https://www.langchain.com/) framework.
        - Filtering involves filtering for only textual data from the repo.
        - Metadata is of 2 levels:
            - Doc Level
                - At the moment this includes things such as filename, path, repo name, length. This can be improved to capture advanced aspects like summary of the entire script and so on.
            - Chunk level
                - At the moment, just has chunk number and length. This can also be improved to capture semantically driven metadata such as function names, comments etc.
                - Chunks are also updated with source info which is inturn used to provide provenance information for the responses.
        - Embedding:
            - Currently the [Salesforce CodeT5 plus 100 embedding model](https://huggingface.co/Salesforce/codet5p-110m-embedding) embedding model is used. The motivation behind this is that the model is light-weight, explicitly trained with github data, trained for text-code alignment, has encoder-decoder, decoder variants etc.
            - The chunked docs are directly embedded by LangChain or there is support for further enrichment of the embeddings to bring about things like Contextual RAG, to infuse the embeddings with richer semantics (such as summary info about the chunk etc) (this is controlled by the `insert_custom_embeddings` flag and following that up with appropriate implementation)
    - Generate (`/generate`)
        - This API takes in the query and retrieves the top-k similar chunks from Qdrant and leverages an LLM to generate a response.
        - Currently OpenAI's GPT 3.5-turbo is used for token generation.
        - This can further be improved by having a self-hosted/self-finetuned OS model such as CodeLLaMa, Cohere etc

- The prompts are managed in the [prompts.toml](/api/configs/prompts.toml) file and other configs are present in [properties.toml](/api/configs/properties.toml)

