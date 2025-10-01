# 🚀 PyPI Deployment Setup Guide for UnicoLab

This guide will help you set up automated PyPI deployment for the KMR package under the UnicoLab organization.

## 📋 Prerequisites

1. **UnicoLab PyPI Account**: Create a PyPI account for UnicoLab organization
2. **GitHub Repository**: Ensure the repository is under `UnicoLab/keras-model-registry`
3. **GitHub Actions**: Enable GitHub Actions in repository settings

## 🔧 Step 1: Create UnicoLab PyPI Account

### 1.1 Register PyPI Account
1. Go to [PyPI Registration](https://pypi.org/account/register/)
2. Use an official UnicoLab email address (e.g., `contact@unicolab.ai`)
3. Choose a username like `unicolab` or `unicolab-org`
4. Complete the registration process

### 1.2 Generate API Token
1. Log into your PyPI account
2. Go to **Account Settings** → **API tokens**
3. Click **Add API token**
4. Choose **Scope**: `Entire account (all projects)`
5. **Token name**: `KMR Package Deployment`
6. Copy the generated token (starts with `pypi-`)

## 🔐 Step 2: Configure GitHub Secrets

### 2.1 Add PyPI Token to GitHub
1. Go to your GitHub repository: `https://github.com/UnicoLab/keras-model-registry`
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. **Name**: `PYPI_API_TOKEN`
5. **Value**: Paste your PyPI API token
6. Click **Add secret**

### 2.2 Verify Secrets
Your repository should now have:
- `PYPI_API_TOKEN`: Your PyPI API token
- `GITHUB_TOKEN`: Automatically provided by GitHub

## 🚀 Step 3: Test Deployment

### 3.1 Manual Deployment (Recommended First)
1. Go to **Actions** tab in your GitHub repository
2. Select **Publish to PyPI - UnicoLab** workflow
3. Click **Run workflow**
4. Enter version: `0.1.0`
5. Click **Run workflow**

### 3.2 Automatic Deployment via Tags
```bash
# Create and push a tag to trigger automatic deployment
git tag v0.1.0
git push origin v0.1.0
```

## 📦 Step 4: Verify Package on PyPI

1. Go to [PyPI](https://pypi.org/project/kmr/)
2. Verify the package appears with UnicoLab branding
3. Test installation:
   ```bash
   pip install kmr
   ```

## 🔄 Step 5: Update Package Information

### 5.1 Package Metadata
The `pyproject.toml` is already configured with:
- **Name**: `kmr`
- **Author**: `UnicoLab <contact@unicolab.ai>`
- **Homepage**: `https://unicolab.ai`
- **Repository**: `https://github.com/UnicoLab/keras-model-registry`
- **Documentation**: `https://unicolab.github.io/keras-model-registry/`

### 5.2 Package Description
The package description includes UnicoLab branding:
```
"Reusable Model Architecture Bricks in Keras - Enterprise AI by UnicoLab"
```

## 🎯 Step 6: Release Process

### 6.1 Version Bumping
Update version in `pyproject.toml`:
```toml
[tool.poetry]
version = "0.1.1"  # Increment as needed
```

### 6.2 Create Release
```bash
# Update version
poetry version patch  # or minor, major

# Create tag
git add pyproject.toml
git commit -m "Bump version to $(poetry version -s)"
git tag v$(poetry version -s)
git push origin main --tags
```

### 6.3 GitHub Release
The workflow automatically creates a GitHub release with:
- Release notes
- Documentation deployment
- PyPI package upload

## 🛠️ Workflow Features

The `PUBLISH_PYPI.yml` workflow includes:

### ✅ Automated Testing
- Runs full test suite before deployment
- Ensures code quality with linting
- Validates package structure

### ✅ Build Process
- Uses Poetry for dependency management
- Builds wheel and source distributions
- Validates package metadata

### ✅ Deployment
- Uploads to PyPI with proper authentication
- Creates GitHub release automatically
- Deploys documentation to GitHub Pages

### ✅ Security
- Uses GitHub secrets for API tokens
- No hardcoded credentials
- Secure token handling

## 🔍 Troubleshooting

### Common Issues

#### 1. Authentication Failed
```
Error: HTTP 403: The user 'unicolab' isn't allowed to upload to project 'kmr'
```
**Solution**: Ensure the PyPI token has the correct scope and the package name is available.

#### 2. Package Already Exists
```
Error: File already exists
```
**Solution**: Increment the version number in `pyproject.toml`.

#### 3. Build Failed
```
Error: Poetry build failed
```
**Solution**: Check `pyproject.toml` syntax and dependencies.

### Debug Steps
1. Check GitHub Actions logs
2. Verify PyPI token permissions
3. Test local build: `poetry build`
4. Validate package: `poetry check`

## 📚 Additional Resources

- [PyPI Documentation](https://packaging.python.org/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [UnicoLab Website](https://unicolab.ai)

## 🎉 Success Checklist

- [ ] UnicoLab PyPI account created
- [ ] API token generated and added to GitHub secrets
- [ ] Package builds successfully locally
- [ ] First deployment to PyPI successful
- [ ] Package appears on PyPI with correct metadata
- [ ] Documentation deployed to GitHub Pages
- [ ] Installation test: `pip install kmr`

## 🚀 Next Steps

After successful deployment:

1. **Announce Release**: Share on UnicoLab social media and blog
2. **Monitor Usage**: Track PyPI download statistics
3. **Gather Feedback**: Collect user feedback for improvements
4. **Plan Next Release**: Prepare roadmap for future versions

---

**Need Help?** Contact the UnicoLab team at [contact@unicolab.ai](mailto:contact@unicolab.ai)
