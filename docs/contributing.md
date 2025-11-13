# Contributing to KerasFactory

Thank you for your interest in contributing to KerasFactory! This guide will help you get started with contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- Git
- Basic knowledge of Keras 3 and deep learning

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/UnicoLab/KerasFactory.git
   cd KerasFactory
   ```

2. **Install Dependencies**
   ```bash
   poetry install
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   make all_tests
   ```

## 📋 Types of Contributions

### 🧩 Adding New Layers

New layers are the core of KerasFactory. Follow these guidelines:

#### Layer Requirements
- **Keras 3 Only**: No TensorFlow dependencies in production code (Tensorflow only for testing)
- **Inherit from BaseLayer**: All layers must inherit from `kerasfactory.layers._base_layer.BaseLayer`
- **Full Serialization**: Implement `get_config()` and `from_config()` methods
- **Type Annotations**: Use Python 3.12+ type hints
- **Comprehensive Documentation**: Google-style docstrings
- **Parameter Validation**: Implement `_validate_params()` method

#### File Structure
- **File Name**: `YourLayer.py` (PascalCase)
- **Location**: `kerasfactory/layers/YourLayer.py`
- **Export**: Add to `kerasfactory/layers/__init__.py`

### 🏗️ Adding New Models

Models should inherit from `kerasfactory.models._base.BaseModel` and follow similar patterns to layers.

### 🧪 Adding Tests

Every layer and model must have comprehensive tests:

#### Test File Structure
- **File Name**: `test__YourLayer.py` (note the double underscore)
- **Location**: `tests/layers/test__YourLayer.py` or `tests/models/test__YourModel.py`

#### Required Tests
- Initialization tests
- Invalid parameter tests
- Build tests
- Output shape tests
- Serialization tests
- Training mode tests
- Model integration tests

## 🔄 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write your code following the guidelines above
- Add comprehensive tests
- Update documentation if needed

### 3. Run Tests
```bash
# Run all tests
make all_tests

# Run specific test file
poetry run python -m pytest tests/layers/test__YourLayer.py -v

# Run with coverage
make coverage
```

### 4. Documentation
Documentation is automatically generated from docstrings using MkDocs and mkdocstrings. Simply ensure your docstrings follow Google style format and the documentation will be updated automatically when the site is built. If you are adding new code, you will have to reference it in a dedicated docuemntation file to fetche the correct docstring.

### 5. Commit Changes
Use conventional commit messages:
```bash
git add .
git commit -m "feat(layers): add YourLayer for feature processing"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 📝 Commit Convention

We use conventional commits:

- `feat(layers): add new layer for feature processing`
- `fix(models): resolve serialization issue in TerminatorModel`
- `docs(readme): update installation instructions`
- `test(layers): add tests for YourLayer`
- `refactor(utils): improve data analyzer performance`

## 🧪 Testing Guidelines

### Test Coverage
- **Minimum 90%**: All new code must have 90%+ test coverage
- **All Paths**: Test both success and failure cases
- **Edge Cases**: Test boundary conditions and edge cases

### Test Categories
1. **Unit Tests**: Individual layer/model functionality
2. **Integration Tests**: Layer combinations and model workflows
3. **Serialization Tests**: Save/load functionality
4. **Performance Tests**: For computationally intensive components

## 🚫 What Not to Include

### Experimental Components
- **Location**: `experimental/` directory (outside package)
- **Purpose**: Research and development
- **Status**: Not included in PyPI package
- **Dependencies**: May use TensorFlow for testing

### Prohibited Dependencies
- **TensorFlow**: Only allowed in test files
- **PyTorch**: Not allowed
- **Other ML Frameworks**: Keras 3 only

## 📞 Getting Help

- **GitHub Issues**: [GitHub Issues](https://github.com/UnicoLab/KerasFactory/issues)
- **GitHub Discussions**: [GitHub Discussions](https://github.com/UnicoLab/KerasFactory/discussions)
- **Documentation**: Check the docs first

## 🎯 Code Review Process

### Review Criteria
1. **Functionality**: Does the code work as intended?
2. **Tests**: Are there comprehensive tests?
3. **Documentation**: Is the code well-documented?
4. **Style**: Does it follow project conventions?
5. **Performance**: Is it efficient?
6. **Security**: Are there any security concerns?

### Review Timeline
- **Initial Review**: Within 48 hours
- **Follow-up**: Within 24 hours of changes
- **Merge**: After approval and CI passes

## 🏆 Recognition

Contributors will be recognized in:
- **README**: Listed as contributors
- **Release Notes**: Mentioned in relevant releases
- **Documentation**: Credited for significant contributions

## 📄 License

By contributing to KerasFactory, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to KerasFactory! Your contributions help make tabular data processing with Keras more accessible and powerful for everyone.
