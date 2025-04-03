# Contributing to LangChain Aerospike

Thank you for your interest in contributing to LangChain Aerospike! This document provides guidelines and instructions for contributing to this project.

## Development Environment Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Prerequisites

- Python 3.8+
- Poetry 1.0.0+

### Installing Poetry

If you don't have Poetry installed, you can install it using:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Make sure it's added to your PATH according to the installation instructions.

### Setting Up the Project

1. Clone the repository:
```bash
git clone https://github.com/aerospike/langchain-aerospike.git
cd langchain-aerospike
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Development Workflow

### Running Tests

```bash
# Run all tests
poetry run test

# Run specific tests
poetry run pytest tests/test_aerospike.py
```

### Formatting Code

We use Black for code formatting:

```bash
poetry run format
```

### Linting

We use Ruff for linting:

```bash
poetry run lint
```

### Adding Dependencies

To add a new dependency:

```bash
# Add a main dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name
```

### Building the Package

To build the package:

```bash
poetry build
```

This will create both a source distribution and a wheel in the `dist/` directory.

### Installing the Package Locally

To install the package locally for testing:

```bash
poetry install
```

## Pull Request Process

1. Create a new branch for your feature or bugfix
2. Implement your changes
3. Add tests for your changes
4. Ensure all tests pass
5. Format your code
6. Submit a pull request

## Publishing to PyPI

This section is for maintainers only.

To publish a new version to PyPI:

1. Update the version in `pyproject.toml`:
```bash
poetry version patch  # or minor, major, etc.
```

2. Create a tag and commit:
```bash
git add pyproject.toml
git commit -m "Bump version to $(poetry version -s)"
git tag v$(poetry version -s)
git push origin main --tags
```

3. Build and publish:
```bash
poetry build
poetry publish
```

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License. 