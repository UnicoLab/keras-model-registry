# ------------------------------------
# Customized func / var define
# ------------------------------------

HAS_POETRY := $(shell command -v poetry 2> /dev/null)
POETRY_VERSION := $(shell poetry version $(shell git describe --tags --abbrev=0))

# ------------------------------------
# Test
# ------------------------------------

.PHONY: unittests
## Run unittests
unittests:
	poetry run python -m pytest

.PHONY: data_analyzer_tests
## Run data analyzer tests
data_analyzer_tests:
	poetry run python -m pytest tests/utils/test_data_analyzer.py tests/utils/test_data_analyzer_cli.py -v

.PHONY: data_analyzer_integration_tests
## Run data analyzer integration tests
data_analyzer_integration_tests:
	poetry run python -m pytest tests/utils/test_data_analyzer_integration.py -v

.PHONY: data_analyzer_all_tests
## Run all data analyzer tests including unit and integration tests
data_analyzer_all_tests:
	poetry run python -m pytest tests/utils/ -v

.PHONY: all_tests
## Run all tests in the project
all_tests:
	poetry run python -m pytest -v

.PHONY: clean_tests
## Remove pytest cache and junit report after tests
clean_tests:
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type f -name '*junit_report.xml' -exec rm {} +

.PHONY: coverage
## Combine and build final coverage for all modules
coverage:
	poetry run coverage run -m pytest
	poetry run coverage combine --data-file .coverage || true
	poetry run coverage html -i
	poetry run coverage report -i

.PHONY: detailed_coverage
## Generate detailed coverage report for the entire project with branch coverage
detailed_coverage:
	poetry run coverage run --branch -m pytest
	poetry run coverage html -i
	poetry run coverage report -i
	@echo "Coverage report generated in: htmlcov/index.html"

.PHONY: data_analyzer_coverage
## Generate coverage report for data analyzer module only
data_analyzer_coverage:
	poetry run coverage run --source=kmr.utils.data_analyzer,kmr.utils.data_analyzer_cli -m pytest tests/utils/
	poetry run coverage report -m

.PHONY: data_analyzer_coverage_detailed
## Generate detailed coverage report for data analyzer with branch coverage
data_analyzer_coverage_detailed:
	poetry run python tests/utils/coverage_report.py

.PHONY: data_analyzer_missing_coverage
## Find and report on missing coverage in data analyzer
data_analyzer_missing_coverage:
	poetry run python scripts/run_missing_coverage.py

# ------------------------------------
# Build package
# ------------------------------------

.PHONY: build_pkg
## Build the package using poetry
build_pkg:
	@echo "Start to build pkg"
ifdef HAS_POETRY
	@$(POETRY_VERSION)
	poetry build
else
	@echo "To build the package, you need to have poetry first"
	exit 1
endif

.PHONY: build
## Clean up cache from previous built, and build the package
build: clean_built build_pkg

.PHONY: clean_built
## Remove cache, built package, and docs directories after build or installation
clean_built:
	find . -type d -name dist -exec rm -r {} +
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# ------------------------------------
# Build doc
# ------------------------------------

.PHONY: docs_deploy
## Build docs using mike
docs_deploy:
	@echo "Starting to build docs"
	@echo "more info: https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/"
ifdef HAS_POETRY
	@$(POETRY_VERSION)
	poetry version -s | xargs -I {} sh -c 'echo Deploying version {} && mike deploy --push --update-aliases {} latest'
else
	@echo "To build the docs, you need to have poetry first"
	exit 1
endif

.PHONY: docs_version_list
## List available versions of the docs
docs_version_list:
	poetry run mike list

.PHONY: docs_version_serve
## Serve versioned docs
docs_version_serve:
	@echo "Start to serve versioned docs"
	poetry run mike serve

.PHONY: docs
## Create or Deploy MkDocs based documentation to GitHub pages.
deploy_doc:
	poetry run mkdocs gh-deploy

.PHONY: serve_doc
## Test MkDocs based documentation locally.
serve_doc:
	poetry run mkdocs serve

# ------------------------------------
# Clean All
# ------------------------------------

.PHONY: clean
## Remove cache, built package, and docs directories after build or installation
clean:
	find . -type d -name dist -exec rm -r {} +
	find . -type f -name '*.rst' ! -name 'index.rst' -delete
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# ------------------------------------
# GitHub Actions Local Testing (Act)
# ------------------------------------

.PHONY: act-install
## Install act tool for local GitHub Actions testing
act-install:
	@echo "Installing act..."
	@command -v act >/dev/null 2>&1 || { \
		echo "Installing act via brew (macOS) or script..."; \
		if command -v brew >/dev/null 2>&1; then \
			brew install act; \
		else \
			curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash; \
		fi \
	}
	@act --version

.PHONY: act-setup
## Set up act configuration (copy secrets template if needed)
act-setup:
	@if [ ! -f .secrets ]; then \
		echo "Creating .secrets from template..."; \
		cp .secrets.example .secrets; \
		echo "Please edit .secrets and add your tokens"; \
	else \
		echo ".secrets already exists"; \
	fi

.PHONY: act-list
## List all available workflows
act-list:
	@act -l

.PHONY: act-test-pr-checks
## Test PR checks workflow locally
act-test-pr-checks:
	@act pull_request -W .github/workflows/pr-checks.yml --eventpath .github/workflows/event-pr.json || echo "Create .github/workflows/event-pr.json for testing"

.PHONY: act-test-tests
## Test tests workflow locally
act-test-tests:
	@act push -W .github/workflows/tests.yml

.PHONY: act-test-docs
## Test docs workflow locally
act-test-docs:
	@act push -W .github/workflows/docs.yml

.PHONY: act-test-pr-preview
## Test PR preview workflow locally
act-test-pr-preview:
	@act pull_request -W .github/workflows/PR_PREVIEW.yml --eventpath .github/workflows/event-pr.json || echo "Create .github/workflows/event-pr.json for testing"

.PHONY: act-test-all
## Test all workflows (dry-run, list only)
act-test-all:
	@echo "Listing all workflows..."
	@act -l

.PHONY: act-test-workflow
## Test a specific workflow (usage: make act-test-workflow WORKFLOW=workflow-name.yml [EVENT=push|pull_request])
act-test-workflow:
	@if [ -z "$(WORKFLOW)" ]; then \
		echo "Usage: make act-test-workflow WORKFLOW=workflow-name.yml [EVENT=push]"; \
		exit 1; \
	fi
	@EVENT=$${EVENT:-push}; \
	echo "Testing workflow $(WORKFLOW) with event $$EVENT..."; \
	act $$EVENT -W .github/workflows/$(WORKFLOW)

.PHONY: act-test-workflow-dry
## Test a specific workflow in dry-run mode (no actual execution)
act-test-workflow-dry:
	@if [ -z "$(WORKFLOW)" ]; then \
		echo "Usage: make act-test-workflow-dry WORKFLOW=workflow-name.yml [EVENT=push]"; \
		exit 1; \
	fi
	@EVENT=$${EVENT:-push}; \
	echo "Dry-running workflow $(WORKFLOW) with event $$EVENT..."; \
	act $$EVENT -W .github/workflows/$(WORKFLOW) --dryrun

.PHONY: act-clean
## Clean act containers and volumes
act-clean:
	@echo "Cleaning act containers..."
	@docker ps -a --filter "ancestor=catthehacker/ubuntu:act-latest" --format "{{.ID}}" | xargs -r docker rm -f || true
	@echo "Cleaning act volumes..."
	@docker volume ls --filter "label=act" --format "{{.Name}}" | xargs -r docker volume rm || true

# ------------------------------------
# Default
# ------------------------------------

.DEFAULT_GOAL := help

help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
