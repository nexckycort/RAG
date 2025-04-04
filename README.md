# RAG Proof of Concept

This project is a Proof of Concept (PoC) for a RAG (Retrieval-Augmented Generation) system that combines a vector database with a language model to answer questions based on stored information.

## Technologies Used
- **FastAPI** for the embedding server
- **SentenceTransformers** with `all-MiniLM-L6-v2` for generating embeddings
- **Pinecone** as the vector database
- **TypeScript** for query processing

## Installation and Setup

### 1. Clone the Repository
```sh
git clone https://github.com/nexckycort/rag-poc.git
cd rag-poc
```

### 2. Install Monorepo Dependencies
```sh
bun install
```

### 3. Set Up the Embedding Server
```sh
cd packages/model-server
bun create:venv  # Create virtual environment
bun activate:venv  # Activate virtual environment
bun pip:install  # Install dependencies
bun dev  # Start server
```

### 4. Run Query Logic
```sh
cd packages/api-server
bun dev  # Run query logic
```

## Usage
1. **Add documents**: Files are processed and stored as embeddings in Pinecone.
2. **Ask a question**: An embedding is generated from the user’s query, and relevant text is retrieved.
3. **Generate answer**: A language model generates a response based on the retrieved context.

## API Endpoints
### Embedding Server (FastAPI)
- `POST /embed` → Generates text embeddings
- `POST /ask` → Queries the model

## License
This project is licensed under the MIT License.
