# ------------------------------------
# Customized func / var define
# ------------------------------------

HAS_POETRY := $(shell command -v poetry 2> /dev/null)

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
	poetry run coverage run --source=kerasfactory.utils.data_analyzer,kerasfactory.utils.data_analyzer_cli -m pytest tests/utils/
	poetry run coverage report -m

.PHONY: data_analyzer_coverage_detailed
## Generate detailed coverage report for data analyzer with branch coverage
data_analyzer_coverage_detailed:
	poetry run python tests/utils/coverage_report.py

.PHONY: data_analyzer_missing_coverage
## Find and report on missing coverage in data analyzer
data_analyzer_missing_coverage:
	@echo "Warning: scripts/run_missing_coverage.py not found. Use data_analyzer_coverage_detailed instead."
	@echo "Running detailed coverage report instead..."
	@$(MAKE) data_analyzer_coverage_detailed

# ------------------------------------
# Build package
# ------------------------------------

.PHONY: build_pkg
## Build the package using poetry
build_pkg:
	@echo "Start to build pkg"
ifdef HAS_POETRY
	@echo "Current version: $$(poetry version -s)"
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
## Build docs using mike (uses poetry version as source of truth, matches PyPI version)
docs_deploy:
	@echo "Starting to build docs"
	@echo "more info: https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/"
ifdef HAS_POETRY
	@VERSION=$$(poetry version -s); \
	echo "Deploying documentation version: $$VERSION (from pyproject.toml - matches PyPI)"; \
	mike delete latest --push 2>/dev/null || true; \
	mike deploy --push --update-aliases "$$VERSION" latest
else
	@echo "To build the docs, you need to have poetry first"
	exit 1
endif

.PHONY: docs_version_check
## Check version consistency between poetry, PyPI, and documentation
docs_version_check:
	@echo "=== Version Consistency Check ==="
	@POETRY_VERSION=$$(poetry version -s); \
	echo "Poetry version (pyproject.toml): $$POETRY_VERSION"; \
	echo ""; \
	echo "Version synchronization flow:"; \
	echo "  1. Semantic-release determines version from commit messages"; \
	echo "  2. Poetry version is updated to match semantic-release version"; \
	echo "  3. pyproject.toml is committed back to repo (keeps it synchronized)"; \
	echo "  4. Package is published to PyPI with this version"; \
	echo "  5. Documentation is deployed with mike using this version"; \
	echo ""; \
	echo "This version should match:"; \
	echo "  - PyPI package version"; \
	echo "  - Documentation version deployed with mike"; \
	echo "  - Git tags (may have 'v' prefix: v$$POETRY_VERSION)"

.PHONY: docs_version_list
## List available versions of the docs
docs_version_list:
	@echo "=== Mike Versions and Aliases ==="
	@poetry run mike list || echo "No versions found or error occurred"
	@echo ""
	@echo "=== Detailed version info from gh-pages branch ==="
	@git fetch origin gh-pages 2>/dev/null || true
	@git show origin/gh-pages:versions.json 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Could not fetch versions.json"
	@echo ""
	@echo "=== Current Poetry Version (should match deployed versions) ==="
	@poetry version -s

.PHONY: docs_version_delete
## Delete a specific version (usage: make docs_version_delete VERSION=latest)
docs_version_delete:
ifdef VERSION
	@echo "Deleting version: $(VERSION)"
	poetry run mike delete $(VERSION) --push
else
	@echo "Error: VERSION not specified. Usage: make docs_version_delete VERSION=latest"
	@exit 1
endif

.PHONY: docs_version_serve
## Serve versioned docs
docs_version_serve:
	@echo "Start to serve versioned docs"
	poetry run mike serve

.PHONY: deploy_doc
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
