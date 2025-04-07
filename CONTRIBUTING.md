# Contributing to langchain-aerospike

Thank you for your interest in contributing to langchain-aerospike! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by the Aerospike Code of Conduct

## Security

- All security vulnerabilities should be reported through GitHub's private vulnerability reporting feature
- Do not disclose security issues publicly until they have been addressed
- See [SECURITY.md](SECURITY.md) for detailed security reporting guidelines

## Development Process

1. **Fork the Repository**
   - Fork the repository to your personal GitHub account
   - Clone your fork locally

2. **Create a Branch**
   - Create a new branch for your feature or bugfix
   - Use descriptive branch names (e.g., `feature/add-new-vector-store` or `fix/connection-timeout`)

3. **Make Changes**
   - Follow the existing code style and patterns
   - Write clear commit messages
   - Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)
   - Include tests for new features and bug fixes
   - Update documentation as needed

4. **Submit a Pull Request**
   - Push your changes to your fork
   - Create a pull request against the main branch
   - Fill out the pull request template
   - Ensure all CI checks pass

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

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file.

## Questions?

If you have any questions about contributing, please:
1. Check the existing documentation
2. Search existing issues
3. Open a new issue if your question hasn't been addressed 