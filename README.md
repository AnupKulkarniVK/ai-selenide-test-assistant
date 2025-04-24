# ai-selenide-test-assistant

## Description
A Retrieval-Augmented Generation (RAG) pipeline and IntelliJ plugin that generates Selenide automated tests from natural-language scenarios, using an example test corpus stored in a GitHub repo.

## Initial Directory Structure
ai-selenide-test-assistant/         # root folder
├── README.md                       # project overview & instructions
├── .gitignore                      # ignore Python caches, model artifacts, etc.
├── ingestion/                      # code for cloning & chunking tests
│   └── ingest.py                   # script to extract and embed test snippets
├── retrieval/                      # code to query vector store
│   └── retrieve.py                 # script to fetch similar examples
├── generation/                     # code to call LLM and generate tests
│   └── generate.py                 # CLI prototype for end-to-end demo
├── plugin/                         # IntelliJ plugin skeleton
│   ├── src/                        # plugin source (Gradle IntelliJ project)
│   └── build.gradle                # plugin build config
├── model/                          # placeholder for model configs & index files
│   ├── embeddings_index.faiss      # FAISS index (to be generated)
│   └── config.json                 # embedding/model parameters
└── docs/                           # design docs, API specs, architecture diagrams
└── architecture.md