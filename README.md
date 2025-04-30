# LangChain Aerospike Vector Store

This package contains the LangChain integration for the [Aerospike Vector Database](https://aerospike.com/products/vector-database/). It provides a VectorStore implementation that uses Aerospike for storing and searching vectors.

## Installation

You can install the package directly from PyPI:

```bash
pip install langchain-aerospike
```

## Documentation

For more advanced usage, including configuring distance metrics, indexes, and other options, please refer to the API documentation and the Aerospike Vector Search client documentation

https://langchain-aerospike.readthedocs.io/en/latest/

https://aerospike-vector-search-python-client.readthedocs.io/en/latest/

## Development Installation

This project uses Poetry for dependency management and packaging. To set up your development environment:

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and install dependencies:
```bash
git clone https://github.com/aerospike/langchain-aerospike.git
cd langchain-aerospike
poetry install
```

3. Activate the virtual environment:
```bash
eval $(poetry env activate)
```

## Requirements

See the pyproject.toml file dependencies section.

## Running Examples

The examples in the `examples/` directory require additional dependencies. You can install them with:

```bash
poetry install --with examples
```

This will install dependencies like `langchain-huggingface` which are used in the example scripts.

## Usage

```python
from langchain_aerospike.vectorstores import Aerospike
from langchain_openai import OpenAIEmbeddings
from aerospike_vector_search import Client, HostPort

# Initialize the Aerospike client
client = Client(seeds=[HostPort(host="localhost", port=3000)])

# Initialize the embeddings model
embedding_model = OpenAIEmbeddings()

# Create an Aerospike vector store
vector_store = Aerospike(
    client=client,
    embedding=embedding_model,
    namespace="test",
    set_name="vectors",
    text_key="text",
    metadata_key="metadata",
    vector_key="vector",
)

# Add documents to the vector store
texts = ["Hello world", "Goodbye world", "Hello and goodbye"]
metadatas = [{"source": "greeting"}, {"source": "farewell"}, {"source": "mixed"}]

vector_store.add_texts(
    texts=texts,
    metadatas=metadatas,
)

# Search for similar documents
query = "Hello there"
docs = vector_store.similarity_search(query)

# Print the results
for doc in docs:
    print(doc.page_content, doc.metadata)
```

## Running Tests

```bash
poetry run pytest tests/
```

> **Note:** Aerospike Vector Search server version 1.1.0 or newer is required to run the integration tests.

## Building Documentation

Documentation is built using Sphinx and hosted on ReadTheDocs. To build the documentation locally:

1. Install the documentation dependencies:
```bash
poetry install --with docs
```

2. Activate the virtual environment:
```bash
poetry shell
```

3. Build the documentation:
```bash
cd docs
make html
```

4. View the documentation in your browser by opening `docs/build/html/index.html`.

## Continuous Integration and Deployment

This project uses GitHub Actions for continuous integration and deployment:

- **Build and Release Workflow**: Automatically builds and creates a GitHub Release with the package wheels when a new tag starting with 'v' is pushed

To create a new release:

1. Update the version in `pyproject.toml`
2. Commit the changes
3. Create and push a new tag:
```bash
git tag v0.1.0  # Use appropriate version
git push origin v0.1.0
```

See the [Release Process](RELEASE.md) document for more details.

## Migration

This package was previously part of LangChain Community Integrations and has been migrated to a standalone package following LangChain's integration guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
