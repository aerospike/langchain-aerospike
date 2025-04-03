"""Pytest configuration for langchain-aerospike tests."""
import pytest


def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "requires(package): mark test as requiring a package"
    )
