.PHONY: all help sync test tests integration_test integration_tests local_server lint format type build

all: help

UV_RUN = uv run --no-sync
TEST_FILE ?= tests/unit_tests/
PYTEST_EXTRA ?=
PYTHON_FILES ?= langchain_aerospike_kv tests
SERVER_ARGS ?=
integration_test integration_tests: TEST_FILE = tests/integration_tests/

sync:
	uv sync --all-groups

test tests:
	$(UV_RUN) --group test pytest --disable-socket --allow-unix-socket $(PYTEST_EXTRA) $(TEST_FILE)

integration_test integration_tests:
	$(UV_RUN) --group test --group test_integration pytest -v --tb=short $(PYTEST_EXTRA) $(TEST_FILE)

local_server:
	$(UV_RUN) --group test --group test_integration python -m tests.local_server $(SERVER_ARGS)

lint:
	$(UV_RUN) --group lint ruff check $(PYTHON_FILES)
	$(UV_RUN) --group lint ruff format --check $(PYTHON_FILES)

format:
	$(UV_RUN) --group lint ruff format $(PYTHON_FILES)
	$(UV_RUN) --group lint ruff check --fix $(PYTHON_FILES)

type:
	$(UV_RUN) --group typing mypy $(PYTHON_FILES)

build:
	$(UV_RUN) --group build python -m build --no-isolation

help:
	@echo 'sync               - explicitly install all development groups'
	@echo 'test               - run unit tests with network sockets disabled'
	@echo 'integration_tests  - run tests against a real Aerospike server'
	@echo 'local_server       - adopt server and run strict live tests (SERVER_ARGS=--create for temporary server)'
	@echo 'lint               - check lint and formatting'
	@echo 'format             - format and fix lint'
	@echo 'type               - run mypy'
	@echo 'build              - build the wheel and sdist without installing'
