"""
Spatial Atlas — Test Configuration

Shared fixtures for all test modules.
"""

import pytest

from config import Config
from llm import LLMClient


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9019",
        help="Base URL of the running agent container",
    )


@pytest.fixture
def agent_url(request):
    return request.config.getoption("--agent-url")


@pytest.fixture
def config():
    """Provide a default config for testing."""
    return Config()


@pytest.fixture
def llm(config):
    """Provide an LLM client (requires API keys in env)."""
    return LLMClient(config)
