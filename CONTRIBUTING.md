# Contributing to LangChain Aerospike

Thank you for your interest in contributing to LangChain Aerospike! This document provides guidelines and instructions for contributing to this project.


## Security

- All security vulnerabilities should be reported through GitHub's private vulnerability reporting feature
- Do not disclose security issues publicly until they have been addressed
- See [SECURITY.md](SECURITY.md) for detailed security reporting guidelines

## Code of Conduct

By participating in this project, you agree to abide by the Aerospike Code of Conduct

## Pull Request Requirements

- All pull requests must be reviewed and approved by at least one maintainer
- All CI checks must pass before merging
- Pull requests from external contributors require approval to run GitHub Actions
- Dependencies must be kept up to date using Dependabot

## Testing

- All new features must include appropriate tests
- Run the test suite locally before submitting a pull request
- Ensure all tests pass before requesting review

## Documentation

- Update relevant documentation when adding new features
- Include clear examples and usage instructions
- Document any breaking changes

## Dependencies

- This project uses Dependabot for dependency management
- All dependency updates are automatically reviewed by the security team
- Major version updates require additional review

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Include docstrings for all public functions and classes
- Keep functions focused and single-purpose

## Pull Request Process

1. **Fork the Repository**
   - Fork the repository to your personal GitHub account
   - Clone your fork locally
2. **Create a Branch**
   - Create a new branch for your feature or bugfix
   - Use descriptive branch names (e.g., `feature/add-new-vector-store` or `fix/connection-timeout`)
3. **Make Changes**
   - Follow the existing code style and patterns
   - Write clear commit messages using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)
   - Include tests for new features and bug fixes
   - Update documentation as needed
4. **Format Your Code
6. **Submit a Pull Request**
   - Push your changes to your fork
   - Create a pull request against the main branch
   - Fill out the pull request template
   - Ensure all CI checks pass

## Questions?

If you have any questions about contributing, please:
1. Check the existing documentation
2. Search existing issues
3. Open a new issue if your question hasn't been addressed 
 

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

By contributing to this project, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file.
