# Local GitHub Actions Testing with Act

This repository supports local testing of GitHub Actions workflows using [act](https://github.com/nektos/act).

## Prerequisites

1. **Install act**: 
   ```bash
   # macOS
   brew install act
   
   # Linux/WSL
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
   
   # Or use the Makefile command
   make act-install
   ```

2. **Set up secrets**:
   ```bash
   make act-setup
   # Then edit .secrets and add your tokens
   ```

3. **Docker**: Ensure Docker is running (act uses Docker to run workflows)

## Quick Start

### List all workflows
```bash
make act-list
# or
act -l
```

### Test a specific workflow
```bash
# Test tests workflow
make act-test-tests

# Test PR checks workflow
make act-test-pr-checks

# Test docs workflow
make act-test-docs

# Test any workflow by name
make act-test-workflow WORKFLOW=tests.yml EVENT=push
```

### Dry-run (see what would run without executing)
```bash
make act-test-workflow-dry WORKFLOW=tests.yml EVENT=push
```

## Available Makefile Commands

- `make act-install` - Install act tool
- `make act-setup` - Set up .secrets file from template
- `make act-list` - List all available workflows
- `make act-test-tests` - Test the tests workflow
- `make act-test-pr-checks` - Test PR checks workflow
- `make act-test-docs` - Test docs workflow
- `make act-test-pr-preview` - Test PR preview workflow
- `make act-test-workflow WORKFLOW=name.yml [EVENT=push]` - Test any workflow
- `make act-test-workflow-dry WORKFLOW=name.yml [EVENT=push]` - Dry-run a workflow
- `make act-clean` - Clean act containers and volumes

## Testing Specific Workflows

### Tests Workflow
```bash
# Test on push event
act push -W .github/workflows/tests.yml

# Test specific job
act push -W .github/workflows/tests.yml -j python-ci-cd
```

### PR Checks Workflow
```bash
# Test PR workflow
act pull_request -W .github/workflows/pr-checks.yml --eventpath .github/workflows/event-pr.json
```

### Documentation Workflow
```bash
act push -W .github/workflows/docs.yml
```

### Release Workflow (requires secrets)
```bash
# Note: Release workflow requires GITHUB_TOKEN and PYPI_TOKEN
act workflow_dispatch -W .github/workflows/RELEASE.yml \
  --input DRY_RUN=true \
  --input SKIP_RELEASE=false
```

## Configuration

### .actrc
The `.actrc` file configures act defaults:
- Platform: `ubuntu-latest=catthehacker/ubuntu:act-latest`
- Secrets file: `.secrets`
- Verbose mode enabled

### .secrets
Create `.secrets` from `.secrets.example` and add your tokens:
- `GITHUB_TOKEN`: GitHub personal access token (repo scope)
- `PYPI_TOKEN`: PyPI API token (for publishing workflows)
- `CODECOV_TOKEN`: Codecov token (for coverage uploads)

## Limitations

⚠️ **Important Notes:**

1. **Reusable Workflows**: Act has limited support for reusable workflows (workflows that use `uses:`). Some workflows that reference `UnicoLab/ul_cicd-workflows` may not run locally.

2. **Composite Actions**: Composite actions are supported but may have limitations.

3. **Self-hosted Actions**: Actions from `.github/templates/` should work locally.

4. **Secrets**: Some workflows may fail locally if required secrets are missing.

5. **GitHub API**: Some workflows that interact with GitHub API may behave differently locally.

## Troubleshooting

### Workflow fails with "reusable workflow not found"
This is expected for workflows using external reusable workflows. Focus on testing:
- Workflows that don't use `uses:` from external repos
- Individual jobs that can be tested in isolation
- Custom composite actions in `.github/templates/`

### Docker permission issues
```bash
# Add your user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in
```

### Clean up act containers
```bash
make act-clean
```

### Verbose output
Act is configured with `-v` flag by default. For more debugging:
```bash
act -v -v -W .github/workflows/tests.yml
```

## Resources

- [Act Documentation](https://github.com/nektos/act#readme)
- [Act User Guide](https://github.com/nektos/act/blob/master/USER_GUIDE.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

