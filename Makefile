.PHONY: help install test lint format type-check check clean docker-build docker-run

help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  type-check   - Run type checking"
	@echo "  check        - Run all checks (lint, type-check, test)"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  run          - Run the chatbot CLI"

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest --cov=rag_chatbot --cov-report=term-missing

lint:
	ruff check src tests
	ruff format --check src tests

format:
	black src tests
	ruff format src tests

type-check:
	mypy src

check: lint type-check test

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t rag-chatbot:latest .

docker-run:
	docker run --rm --env-file .env -it rag-chatbot:latest interactive

run:
	python -m rag_chatbot interactive
