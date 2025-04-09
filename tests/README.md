# Tests for langchain-aerospike

This directory contains tests for the langchain-aerospike package.

## Directory Structure

Tests are organized by component to match the package structure:

- `tests/vectorstores/`: Tests for vectorstore implementations
- Additional component directories will be added as needed

## Common Test Utilities

- `fake_embeddings.py`: Contains the `FakeEmbeddings` class used for testing
- `conftest.py`: Contains pytest fixtures and configuration

## Running Tests

You can run all tests with:

```bash
poetry run test
```

Or run specific component tests:

```bash
poetry run pytest tests/vectorstores
```

> **Important:** Aerospike Vector Search server version 1.1.0 or newer is required to run the integration tests.
