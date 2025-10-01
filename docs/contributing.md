[38;5;243m   1[0m [38;5;249m# Contributing to KMR[0m
[38;5;243m   2[0m 
[38;5;243m   3[0m [38;5;249mThank you for your interest in contributing to the Keras Model Registry (KMR)! This guide will help you get started with contributing to the project.[0m
[38;5;243m   4[0m 
[38;5;243m   5[0m [38;5;249m## 🚀 Getting Started[0m
[38;5;243m   6[0m 
[38;5;243m   7[0m [38;5;249m### Prerequisites[0m
[38;5;243m   8[0m 
[38;5;243m   9[0m [38;5;249m- Python 3.9+[0m
[38;5;243m  10[0m [38;5;249m- Poetry (for dependency management)[0m
[38;5;243m  11[0m [38;5;249m- Git[0m
[38;5;243m  12[0m [38;5;249m- Basic knowledge of Keras 3 and deep learning[0m
[38;5;243m  13[0m 
[38;5;243m  14[0m [38;5;249m### Development Setup[0m
[38;5;243m  15[0m 
[38;5;243m  16[0m [38;5;249m1. **Fork and Clone the Repository**[0m
[38;5;243m  17[0m [38;5;249m   ```bash[0m
[38;5;243m  18[0m [38;5;249m   git clone https://github.com/UnicoLab/keras-model-registry.git[0m
[38;5;243m  19[0m [38;5;249m   cd keras-model-registry[0m
[38;5;243m  20[0m [38;5;249m   ```[0m
[38;5;243m  21[0m 
[38;5;243m  22[0m [38;5;249m2. **Install Dependencies**[0m
[38;5;243m  23[0m [38;5;249m   ```bash[0m
[38;5;243m  24[0m [38;5;249m   poetry install[0m
[38;5;243m  25[0m [38;5;249m   ```[0m
[38;5;243m  26[0m 
[38;5;243m  27[0m [38;5;249m3. **Install Pre-commit Hooks**[0m
[38;5;243m  28[0m [38;5;249m   ```bash[0m
[38;5;243m  29[0m [38;5;249m   pre-commit install[0m
[38;5;243m  30[0m [38;5;249m   ```[0m
[38;5;243m  31[0m 
[38;5;243m  32[0m [38;5;249m4. **Run Tests**[0m
[38;5;243m  33[0m [38;5;249m   ```bash[0m
[38;5;243m  34[0m [38;5;249m   make all_tests[0m
[38;5;243m  35[0m [38;5;249m   ```[0m
[38;5;243m  36[0m 
[38;5;243m  37[0m [38;5;249m## 📋 Types of Contributions[0m
[38;5;243m  38[0m 
[38;5;243m  39[0m [38;5;249m### 🧩 Adding New Layers[0m
[38;5;243m  40[0m 
[38;5;243m  41[0m [38;5;249mNew layers are the core of KMR. Follow these guidelines:[0m
[38;5;243m  42[0m 
[38;5;243m  43[0m [38;5;249m#### Layer Requirements[0m
[38;5;243m  44[0m [38;5;249m- **Keras 3 Only**: No TensorFlow dependencies in production code[0m
[38;5;243m  45[0m [38;5;249m- **Inherit from BaseLayer**: All layers must inherit from `kmr.layers._base_layer.BaseLayer`[0m
[38;5;243m  46[0m [38;5;249m- **Full Serialization**: Implement `get_config()` and `from_config()` methods[0m
[38;5;243m  47[0m [38;5;249m- **Type Annotations**: Use Python 3.12+ type hints[0m
[38;5;243m  48[0m [38;5;249m- **Comprehensive Documentation**: Google-style docstrings[0m
[38;5;243m  49[0m [38;5;249m- **Parameter Validation**: Implement `_validate_params()` method[0m
[38;5;243m  50[0m 
[38;5;243m  51[0m [38;5;249m#### File Structure[0m
[38;5;243m  52[0m [38;5;249m- **File Name**: `YourLayer.py` (PascalCase)[0m
[38;5;243m  53[0m [38;5;249m- **Location**: `kmr/layers/YourLayer.py`[0m
[38;5;243m  54[0m [38;5;249m- **Export**: Add to `kmr/layers/__init__.py`[0m
[38;5;243m  55[0m 
[38;5;243m  56[0m [38;5;249m### 🏗️ Adding New Models[0m
[38;5;243m  57[0m 
[38;5;243m  58[0m [38;5;249mModels should inherit from `kmr.models._base.BaseModel` and follow similar patterns to layers.[0m
[38;5;243m  59[0m 
[38;5;243m  60[0m [38;5;249m### 🧪 Adding Tests[0m
[38;5;243m  61[0m 
[38;5;243m  62[0m [38;5;249mEvery layer and model must have comprehensive tests:[0m
[38;5;243m  63[0m 
[38;5;243m  64[0m [38;5;249m#### Test File Structure[0m
[38;5;243m  65[0m [38;5;249m- **File Name**: `test__YourLayer.py` (note the double underscore)[0m
[38;5;243m  66[0m [38;5;249m- **Location**: `tests/layers/test__YourLayer.py` or `tests/models/test__YourModel.py`[0m
[38;5;243m  67[0m 
[38;5;243m  68[0m [38;5;249m#### Required Tests[0m
[38;5;243m  69[0m [38;5;249m- Initialization tests[0m
[38;5;243m  70[0m [38;5;249m- Invalid parameter tests[0m
[38;5;243m  71[0m [38;5;249m- Build tests[0m
[38;5;243m  72[0m [38;5;249m- Output shape tests[0m
[38;5;243m  73[0m [38;5;249m- Serialization tests[0m
[38;5;243m  74[0m [38;5;249m- Training mode tests[0m
[38;5;243m  75[0m [38;5;249m- Model integration tests[0m
[38;5;243m  76[0m 
[38;5;243m  77[0m [38;5;249m## 🔄 Development Workflow[0m
[38;5;243m  78[0m 
[38;5;243m  79[0m [38;5;249m### 1. Create a Feature Branch[0m
[38;5;243m  80[0m [38;5;249m```bash[0m
[38;5;243m  81[0m [38;5;249mgit checkout -b feature/your-feature-name[0m
[38;5;243m  82[0m [38;5;249m```[0m
[38;5;243m  83[0m 
[38;5;243m  84[0m [38;5;249m### 2. Make Changes[0m
[38;5;243m  85[0m [38;5;249m- Write your code following the guidelines above[0m
[38;5;243m  86[0m [38;5;249m- Add comprehensive tests[0m
[38;5;243m  87[0m [38;5;249m- Update documentation if needed[0m
[38;5;243m  88[0m 
[38;5;243m  89[0m [38;5;249m### 3. Run Tests[0m
[38;5;243m  90[0m [38;5;249m```bash[0m
[38;5;243m  91[0m [38;5;249m# Run all tests[0m
[38;5;243m  92[0m [38;5;249mmake all_tests[0m
[38;5;243m  93[0m 
[38;5;243m  94[0m [38;5;249m# Run specific test file[0m
[38;5;243m  95[0m [38;5;249mpoetry run python -m pytest tests/layers/test__YourLayer.py -v[0m
[38;5;243m  96[0m 
[38;5;243m  97[0m [38;5;249m# Run with coverage[0m
[38;5;243m  98[0m [38;5;249mmake coverage[0m
[38;5;243m  99[0m [38;5;249m```[0m
[38;5;243m 100[0m 
[38;5;243m 101[0m [38;5;249m### 4. Update Documentation[0m
[38;5;243m 102[0m [38;5;249m```bash[0m
[38;5;243m 103[0m [38;5;249mpython scripts/generate_docs.py[0m
[38;5;243m 104[0m [38;5;249m```[0m
[38;5;243m 105[0m 
[38;5;243m 106[0m [38;5;249m### 5. Commit Changes[0m
[38;5;243m 107[0m [38;5;249mUse conventional commit messages:[0m
[38;5;243m 108[0m [38;5;249m```bash[0m
[38;5;243m 109[0m [38;5;249mgit add .[0m
[38;5;243m 110[0m [38;5;249mgit commit -m "feat(layers): add YourLayer for feature processing"[0m
[38;5;243m 111[0m [38;5;249m```[0m
[38;5;243m 112[0m 
[38;5;243m 113[0m [38;5;249m### 6. Push and Create Pull Request[0m
[38;5;243m 114[0m [38;5;249m```bash[0m
[38;5;243m 115[0m [38;5;249mgit push origin feature/your-feature-name[0m
[38;5;243m 116[0m [38;5;249m```[0m
[38;5;243m 117[0m 
[38;5;243m 118[0m [38;5;249m## 📝 Commit Convention[0m
[38;5;243m 119[0m 
[38;5;243m 120[0m [38;5;249mWe use conventional commits:[0m
[38;5;243m 121[0m 
[38;5;243m 122[0m [38;5;249m- `feat(layers): add new layer for feature processing`[0m
[38;5;243m 123[0m [38;5;249m- `fix(models): resolve serialization issue in TerminatorModel`[0m
[38;5;243m 124[0m [38;5;249m- `docs(readme): update installation instructions`[0m
[38;5;243m 125[0m [38;5;249m- `test(layers): add tests for YourLayer`[0m
[38;5;243m 126[0m [38;5;249m- `refactor(utils): improve data analyzer performance`[0m
[38;5;243m 127[0m 
[38;5;243m 128[0m [38;5;249m## 🧪 Testing Guidelines[0m
[38;5;243m 129[0m 
[38;5;243m 130[0m [38;5;249m### Test Coverage[0m
[38;5;243m 131[0m [38;5;249m- **Minimum 90%**: All new code must have 90%+ test coverage[0m
[38;5;243m 132[0m [38;5;249m- **All Paths**: Test both success and failure cases[0m
[38;5;243m 133[0m [38;5;249m- **Edge Cases**: Test boundary conditions and edge cases[0m
[38;5;243m 134[0m 
[38;5;243m 135[0m [38;5;249m### Test Categories[0m
[38;5;243m 136[0m [38;5;249m1. **Unit Tests**: Individual layer/model functionality[0m
[38;5;243m 137[0m [38;5;249m2. **Integration Tests**: Layer combinations and model workflows[0m
[38;5;243m 138[0m [38;5;249m3. **Serialization Tests**: Save/load functionality[0m
[38;5;243m 139[0m [38;5;249m4. **Performance Tests**: For computationally intensive components[0m
[38;5;243m 140[0m 
[38;5;243m 141[0m [38;5;249m## 🚫 What Not to Include[0m
[38;5;243m 142[0m 
[38;5;243m 143[0m [38;5;249m### Experimental Components[0m
[38;5;243m 144[0m [38;5;249m- **Location**: `experimental/` directory (outside package)[0m
[38;5;243m 145[0m [38;5;249m- **Purpose**: Research and development[0m
[38;5;243m 146[0m [38;5;249m- **Status**: Not included in PyPI package[0m
[38;5;243m 147[0m [38;5;249m- **Dependencies**: May use TensorFlow for testing[0m
[38;5;243m 148[0m 
[38;5;243m 149[0m [38;5;249m### Prohibited Dependencies[0m
[38;5;243m 150[0m [38;5;249m- **TensorFlow**: Only allowed in test files[0m
[38;5;243m 151[0m [38;5;249m- **PyTorch**: Not allowed[0m
[38;5;243m 152[0m [38;5;249m- **Other ML Frameworks**: Keras 3 only[0m
[38;5;243m 153[0m 
[38;5;243m 154[0m [38;5;249m## 📞 Getting Help[0m
[38;5;243m 155[0m 
[38;5;243m 156[0m [38;5;249m- **GitHub Issues**: [GitHub Issues](https://github.com/UnicoLab/keras-model-registry/issues)[0m
[38;5;243m 157[0m [38;5;249m- **GitHub Discussions**: [GitHub Discussions](https://github.com/UnicoLab/keras-model-registry/discussions)[0m
[38;5;243m 158[0m [38;5;249m- **Documentation**: Check the docs first[0m
[38;5;243m 159[0m 
[38;5;243m 160[0m [38;5;249m## 🎯 Code Review Process[0m
[38;5;243m 161[0m 
[38;5;243m 162[0m [38;5;249m### Review Criteria[0m
[38;5;243m 163[0m [38;5;249m1. **Functionality**: Does the code work as intended?[0m
[38;5;243m 164[0m [38;5;249m2. **Tests**: Are there comprehensive tests?[0m
[38;5;243m 165[0m [38;5;249m3. **Documentation**: Is the code well-documented?[0m
[38;5;243m 166[0m [38;5;249m4. **Style**: Does it follow project conventions?[0m
[38;5;243m 167[0m [38;5;249m5. **Performance**: Is it efficient?[0m
[38;5;243m 168[0m [38;5;249m6. **Security**: Are there any security concerns?[0m
[38;5;243m 169[0m 
[38;5;243m 170[0m [38;5;249m### Review Timeline[0m
[38;5;243m 171[0m [38;5;249m- **Initial Review**: Within 48 hours[0m
[38;5;243m 172[0m [38;5;249m- **Follow-up**: Within 24 hours of changes[0m
[38;5;243m 173[0m [38;5;249m- **Merge**: After approval and CI passes[0m
[38;5;243m 174[0m 
[38;5;243m 175[0m [38;5;249m## 🏆 Recognition[0m
[38;5;243m 176[0m 
[38;5;243m 177[0m [38;5;249mContributors will be recognized in:[0m
[38;5;243m 178[0m [38;5;249m- **README**: Listed as contributors[0m
[38;5;243m 179[0m [38;5;249m- **Release Notes**: Mentioned in relevant releases[0m
[38;5;243m 180[0m [38;5;249m- **Documentation**: Credited for significant contributions[0m
[38;5;243m 181[0m 
[38;5;243m 182[0m [38;5;249m## 📄 License[0m
[38;5;243m 183[0m 
[38;5;243m 184[0m [38;5;249mBy contributing to KMR, you agree that your contributions will be licensed under the MIT License.[0m
[38;5;243m 185[0m 
[38;5;243m 186[0m [38;5;249m---[0m
[38;5;243m 187[0m 
[38;5;243m 188[0m [38;5;249mThank you for contributing to KMR! Your contributions help make tabular data processing with Keras more accessible and powerful for everyone.[0m
