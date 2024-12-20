# Advanced Retrieval-Augmented Generation (RAG) with Llama3

This repository implements an advanced Retrieval-Augmented Generation (RAG) pipeline using Llama3 and supporting libraries like LangChain. The project demonstrates document parsing, embedding generation, and retrieval-based question-answering with context compression techniques.

## Features
- **Document Parsing**: Utilize `UnstructuredMarkdownLoader` for ingesting markdown documents.
- **Embeddings**: Leverage `FastEmbedEmbeddings` for efficient vector embeddings.
- **Vector Stores**: Store and retrieve embeddings using `Qdrant`.
- **Retrieval QA**: A custom `RetrievalQA` chain with Llama3 as the base LLM.
- **Context Compression**: Use `ContextualCompressionRetriever` and `FlashrankRerank` for efficient retrieval.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip

### Required Libraries
Install the dependencies using:

```bash
pip install -r requirements.txt
```

Alternatively, install key libraries manually:

```bash
pip install langchain llama_parse langchain_community fastembed qdrant-client
```

## Usage

### 1. Parse Documents

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("/path/to/markdown/file.md")
loaded_documents = loader.load()
```

### 2. Generate Embeddings

```python
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

embedder = FastEmbedEmbeddings()
embeddings = embedder.embed_documents([doc.text for doc in loaded_documents])
```

### 3. Store Embeddings in Qdrant

```python
from langchain.vectorstores import Qdrant

qdrant = Qdrant.from_embeddings(embeddings, "example_collection")
```

### 4. Retrieval and QA

```python
from langchain.chains import RetrievalQA

retriever = qdrant.as_retriever()
retrieval_qa = RetrievalQA.from_chain_type(llm=ChatGroq(api_key="YOUR_API_KEY"), retriever=retriever)

response = retrieval_qa.run("Your query here")
print(response)
```

## File Structure

```
project/
├── advanced_RAG_with_llama3.ipynb  # Main Jupyter notebook
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
└── src/                            # Source code (if modularized)
    ├── loaders.py                  # Document loader utilities
    ├── embeddings.py               # Embedding generation
    ├── retriever.py                # Retrieval pipeline
    └── qa_chain.py                 # Retrieval QA implementation
```

## Dependencies

The following key libraries are used:
- **LangChain**: Framework for building LLM-based applications.
- **LlamaParse**: Document parsing utility.
- **Qdrant**: High-performance vector search engine.
- **FastEmbedEmbeddings**: Efficient embedding generator.

## Known Issues
- Ensure NLTK resources (`punkt`, `averaged_perceptron_tagger`) are downloaded:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Create a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
