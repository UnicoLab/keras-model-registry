[38;5;243m   1[0m [38;5;249m# 🤖 Model Implementation Guide for KerasFactory[0m
[38;5;243m   2[0m 
[38;5;243m   3[0m [38;5;249mThis guide outlines the complete process and best practices for implementing new models in the KerasFactory project. Follow the checklists to ensure your implementation meets all KerasFactory standards.[0m
[38;5;243m   4[0m 
[38;5;243m   5[0m [38;5;249m## 📋 Model Implementation Checklist[0m
[38;5;243m   6[0m 
[38;5;243m   7[0m [38;5;249mUse this checklist when implementing a new model. Check off each item as you complete it.[0m
[38;5;243m   8[0m 
[38;5;243m   9[0m [38;5;249m### Phase 1: Planning & Design[0m
[38;5;243m  10[0m [38;5;249m- [ ] **Define Purpose**: Clearly document what the model does and when to use it[0m
[38;5;243m  11[0m [38;5;249m- [ ] **Review Architecture**: Design the model architecture (layers, connections, data flow)[0m
[38;5;243m  12[0m [38;5;249m- [ ] **Plan Layers**: Identify which layers the model needs[0m
[38;5;243m  13[0m [38;5;249m  - [ ] Check if all required layers exist in `kerasfactory/layers/`[0m
[38;5;243m  14[0m [38;5;249m  - [ ] Plan to implement missing layers separately first[0m
[38;5;243m  15[0m [38;5;249m  - [ ] Prioritize reusability (create standalone layers, not embedded logic)[0m
[38;5;243m  16[0m [38;5;249m- [ ] **Define Inputs/Outputs**: Plan input and output specifications[0m
[38;5;243m  17[0m [38;5;249m- [ ] **Document Algorithm**: Write mathematical description or pseudo-code[0m
[38;5;243m  18[0m 
[38;5;243m  19[0m [38;5;249m### Phase 2: Layer Implementation (if needed)[0m
[38;5;243m  20[0m [38;5;249m**IMPORTANT**: Implement any missing layers FIRST as standalone, reusable components.[0m
[38;5;243m  21[0m 
[38;5;243m  22[0m [38;5;249mFor each missing layer:[0m
[38;5;243m  23[0m [38;5;249m- [ ] Follow the **Layer Implementation Checklist** from `layers_implementation_guide.md`[0m
[38;5;243m  24[0m [38;5;249m- [ ] Implement layer code[0m
[38;5;243m  25[0m [38;5;249m- [ ] Write comprehensive tests[0m
[38;5;243m  26[0m [38;5;249m- [ ] Create documentation[0m
[38;5;243m  27[0m [38;5;249m- [ ] Update API references[0m
[38;5;243m  28[0m [38;5;249m- [ ] Run all tests and linting[0m
[38;5;243m  29[0m 
[38;5;243m  30[0m [38;5;249m### Phase 3: Implementation - Core Model Code[0m
[38;5;243m  31[0m [38;5;249m- [ ] **Create File**: Create `kerasfactory/models/YourModelName.py` following naming conventions[0m
[38;5;243m  32[0m [38;5;249m- [ ] **Add Module Docstring**: Document the module's purpose[0m
[38;5;243m  33[0m [38;5;249m- [ ] **Implement Pure Keras 3**: Use only Keras operations (no TensorFlow)[0m
[38;5;243m  34[0m [38;5;249m- [ ] **Apply @register_keras_serializable**: Decorate class with `@register_keras_serializable(package="kerasfactory.models")`[0m
[38;5;243m  35[0m [38;5;249m- [ ] **Inherit from BaseModel**: Extend `kerasfactory.models._base.BaseModel`[0m
[38;5;243m  36[0m [38;5;249m- [ ] **Implement __init__**: [0m
[38;5;243m  37[0m [38;5;249m  - [ ] Set private attributes first (`self._param = param`)[0m
[38;5;243m  38[0m [38;5;249m  - [ ] Validate parameters (in __init__ or _validate_params)[0m
[38;5;243m  39[0m [38;5;249m  - [ ] Set public attributes (`self.param = self._param`)[0m
[38;5;243m  40[0m [38;5;249m  - [ ] Call `super().__init__(name=name, **kwargs)` AFTER setting public attributes[0m
[38;5;243m  41[0m [38;5;249m- [ ] **Implement _validate_params**: Add parameter validation logic[0m
[38;5;243m  42[0m [38;5;249m- [ ] **Implement build()**: Initialize all layers and sublayers[0m
[38;5;243m  43[0m [38;5;249m- [ ] **Implement call()**: Implement forward pass with Keras operations only[0m
[38;5;243m  44[0m [38;5;249m- [ ] **Implement get_config()**: Return all constructor parameters[0m
[38;5;243m  45[0m [38;5;249m- [ ] **Add Type Hints**: All methods and parameters have proper type annotations[0m
[38;5;243m  46[0m [38;5;249m- [ ] **Add Logging**: Use `loguru` for debug messages[0m
[38;5;243m  47[0m [38;5;249m- [ ] **Add Comprehensive Docstring**: Google-style docstring with:[0m
[38;5;243m  48[0m [38;5;249m  - [ ] Description[0m
[38;5;243m  49[0m [38;5;249m  - [ ] Parameters[0m
[38;5;243m  50[0m [38;5;249m  - [ ] Input/output shapes[0m
[38;5;243m  51[0m [38;5;249m  - [ ] Usage examples[0m
[38;5;243m  52[0m [38;5;249m  - [ ] References (if applicable)[0m
[38;5;243m  53[0m 
[38;5;243m  54[0m [38;5;249m### Phase 4: Unit Tests[0m
[38;5;243m  55[0m [38;5;249m- [ ] **Create Test File**: Create `tests/models/test__YourModelName.py`[0m
[38;5;243m  56[0m [38;5;249m- [ ] **Test Initialization**: [0m
[38;5;243m  57[0m [38;5;249m  - [ ] Default parameters[0m
[38;5;243m  58[0m [38;5;249m  - [ ] Custom parameters[0m
[38;5;243m  59[0m [38;5;249m  - [ ] Invalid parameters (should raise errors)[0m
[38;5;243m  60[0m [38;5;249m- [ ] **Test Model Building**: Build with different input shapes[0m
[38;5;243m  61[0m [38;5;249m- [ ] **Test Output Shape**: Verify output shapes match expected values[0m
[38;5;243m  62[0m [38;5;249m- [ ] **Test Output Type**: Verify output is correct dtype[0m
[38;5;243m  63[0m [38;5;249m- [ ] **Test Different Batch Sizes**: Test with various batch dimensions[0m
[38;5;243m  64[0m [38;5;249m- [ ] **Test Forward Pass**: Model produces valid outputs[0m
[38;5;243m  65[0m [38;5;249m- [ ] **Test Training Loop**:[0m
[38;5;243m  66[0m [38;5;249m  - [ ] Can compile the model[0m
[38;5;243m  67[0m [38;5;249m  - [ ] Can train for multiple epochs[0m
[38;5;243m  68[0m [38;5;249m  - [ ] Loss decreases over training[0m
[38;5;243m  69[0m [38;5;249m- [ ] **Test Serialization**:[0m
[38;5;243m  70[0m [38;5;249m  - [ ] `get_config()` returns correct dict[0m
[38;5;243m  71[0m [38;5;249m  - [ ] `from_config()` recreates model correctly[0m
[38;5;243m  72[0m [38;5;249m  - [ ] `keras.saving.serialize_keras_object()` works[0m
[38;5;243m  73[0m [38;5;249m  - [ ] `keras.saving.deserialize_keras_object()` works[0m
[38;5;243m  74[0m [38;5;249m  - [ ] Model can be saved/loaded (`.keras` format)[0m
[38;5;243m  75[0m [38;5;249m  - [ ] Weights can be saved/loaded (`.h5` format)[0m
[38;5;243m  76[0m [38;5;249m  - [ ] Predictions consistent after loading[0m
[38;5;243m  77[0m [38;5;249m- [ ] **Test Deterministic Output**: Same input produces same output (with same seed)[0m
[38;5;243m  78[0m [38;5;249m- [ ] **Test Layer Integration**: All constituent layers work correctly together[0m
[38;5;243m  79[0m [38;5;249m- [ ] **Test Prediction**: Model can make predictions on new data[0m
[38;5;243m  80[0m [38;5;249m- [ ] **All Tests Pass**: Run `pytest tests/models/test__YourModelName.py -v`[0m
[38;5;243m  81[0m 
[38;5;243m  82[0m [38;5;249m### Phase 5: Documentation[0m
[38;5;243m  83[0m [38;5;249m- [ ] **Create Documentation File**: Create `docs/models/your-model-name.md`[0m
[38;5;243m  84[0m [38;5;249m- [ ] **Follow Template**: Use structure from similar model in `docs/models/`[0m
[38;5;243m  85[0m [38;5;249m- [ ] **Include Comprehensive Sections**:[0m
[38;5;243m  86[0m [38;5;249m  - [ ] Overview and problem it solves[0m
[38;5;243m  87[0m [38;5;249m  - [ ] Architecture overview with diagram[0m
[38;5;243m  88[0m [38;5;249m  - [ ] Key features and innovations[0m
[38;5;243m  89[0m [38;5;249m  - [ ] Input/output specifications[0m
[38;5;243m  90[0m [38;5;249m  - [ ] Parameters and their impact[0m
[38;5;243m  91[0m [38;5;249m  - [ ] Quick start example[0m
[38;5;243m  92[0m [38;5;249m  - [ ] Advanced usage (custom training loop, transfer learning, etc.)[0m
[38;5;243m  93[0m [38;5;249m  - [ ] Performance characteristics and benchmarks[0m
[38;5;243m  94[0m [38;5;249m  - [ ] Comparison with related architectures[0m
[38;5;243m  95[0m [38;5;249m  - [ ] Training best practices[0m
[38;5;243m  96[0m [38;5;249m  - [ ] Common issues & troubleshooting[0m
[38;5;243m  97[0m [38;5;249m  - [ ] Integration with other KerasFactory components[0m
[38;5;243m  98[0m [38;5;249m  - [ ] References and citations[0m
[38;5;243m  99[0m [38;5;249m- [ ] **Add Code Examples**: Real, working examples (training, evaluation, prediction)[0m
[38;5;243m 100[0m [38;5;249m- [ ] **Include Mathematical Details**: Equations, loss functions, optimization details[0m
[38;5;243m 101[0m [38;5;249m- [ ] **Add Visual Aids**: Architecture diagrams, Mermaid diagrams, flowcharts[0m
[38;5;243m 102[0m [38;5;249m- [ ] **Include Reproducibility Info**: Random seeds, hardware requirements, etc.[0m
[38;5;243m 103[0m 
[38;5;243m 104[0m [38;5;249m### Phase 6: Jupyter Notebook Example[0m
[38;5;243m 105[0m [38;5;249m- [ ] **Create Notebook**: Create `notebooks/your_model_name_demo.ipynb` or `your_model_name_end_to_end_demo.ipynb`[0m
[38;5;243m 106[0m [38;5;249m- [ ] **Include Sections**:[0m
[38;5;243m 107[0m [38;5;249m  - [ ] Title and description[0m
[38;5;243m 108[0m [38;5;249m  - [ ] Setup and imports[0m
[38;5;243m 109[0m [38;5;249m  - [ ] Data generation/loading[0m
[38;5;243m 110[0m [38;5;249m  - [ ] Data exploration/visualization[0m
[38;5;243m 111[0m [38;5;249m  - [ ] Model creation and architecture overview[0m
[38;5;243m 112[0m [38;5;249m  - [ ] Model training with visualization[0m
[38;5;243m 113[0m [38;5;249m  - [ ] Model evaluation[0m
[38;5;243m 114[0m [38;5;249m  - [ ] Predictions and visualization[0m
[38;5;243m 115[0m [38;5;249m  - [ ] Performance comparison (if applicable)[0m
[38;5;243m 116[0m [38;5;249m  - [ ] Model serialization and loading[0m
[38;5;243m 117[0m [38;5;249m  - [ ] Best practices and tips[0m
[38;5;243m 118[0m [38;5;249m  - [ ] Summary and conclusions[0m
[38;5;243m 119[0m [38;5;249m- [ ] **Add Visualizations**: [0m
[38;5;243m 120[0m [38;5;249m  - [ ] Training curves (loss, metrics)[0m
[38;5;243m 121[0m [38;5;249m  - [ ] Predictions vs actual[0m
[38;5;243m 122[0m [38;5;249m  - [ ] Performance metrics[0m
[38;5;243m 123[0m [38;5;249m  - [ ] Model comparisons (if applicable)[0m
[38;5;243m 124[0m [38;5;249m- [ ] **Include Output**: Run all cells to verify they work[0m
[38;5;243m 125[0m [38;5;249m- [ ] **Use Interactive Plots**: Plotly for better interactivity[0m
[38;5;243m 126[0m 
[38;5;243m 127[0m [38;5;249m### Phase 7: Integration & Updates[0m
[38;5;243m 128[0m [38;5;249m- [ ] **Update Imports**: Add to `kerasfactory/models/__init__.py`[0m
[38;5;243m 129[0m [38;5;249m  - [ ] Add import statement[0m
[38;5;243m 130[0m [38;5;249m  - [ ] Add model name to `__all__` list[0m
[38;5;243m 131[0m [38;5;249m- [ ] **Update API Documentation**: Add entry to `docs/api/models.md`[0m
[38;5;243m 132[0m [38;5;249m  - [ ] Add model name and description[0m
[38;5;243m 133[0m [38;5;249m  - [ ] Include autodoc reference (`::: kerasfactory.models.YourModelName`)[0m
[38;5;243m 134[0m [38;5;249m  - [ ] List key features[0m
[38;5;243m 135[0m [38;5;249m  - [ ] Add use case recommendations[0m
[38;5;243m 136[0m [38;5;249m- [ ] **Update Models Overview**: If exists, add to `docs/models_overview.md` or similar[0m
[38;5;243m 137[0m [38;5;249m- [ ] **Update Main README**: If it's a significant model[0m
[38;5;243m 138[0m [38;5;249m  - [ ] Add to feature list[0m
[38;5;243m 139[0m [38;5;249m  - [ ] Link to documentation[0m
[38;5;243m 140[0m [38;5;249m- [ ] **Update Tutorials**: If introducing new concepts[0m
[38;5;243m 141[0m [38;5;249m- [ ] **Update Data Analyzer**: If applicable, add to `kerasfactory/utils/data_analyzer.py`[0m
[38;5;243m 142[0m 
[38;5;243m 143[0m [38;5;249m### Phase 8: Quality Assurance[0m
[38;5;243m 144[0m [38;5;249m- [ ] **Run All Tests**: [0m
[38;5;243m 145[0m [38;5;249m  - [ ] Model tests pass: `pytest tests/models/test__YourModelName.py -v`[0m
[38;5;243m 146[0m [38;5;249m  - [ ] All layer tests pass: `pytest tests/layers/ -v`[0m
[38;5;243m 147[0m [38;5;249m  - [ ] No regressions: `pytest tests/ -v`[0m
[38;5;243m 148[0m [38;5;249m- [ ] **Pre-commit Hooks**: Run `pre-commit run --all-files`[0m
[38;5;243m 149[0m [38;5;249m  - [ ] Black formatting passes[0m
[38;5;243m 150[0m [38;5;249m  - [ ] Ruff linting passes[0m
[38;5;243m 151[0m [38;5;249m  - [ ] No unused imports or variables[0m
[38;5;243m 152[0m [38;5;249m  - [ ] Proper type hints[0m
[38;5;243m 153[0m [38;5;249m  - [ ] Docstring formatting[0m
[38;5;243m 154[0m [38;5;249m- [ ] **Documentation Build**: `mkdocs serve` builds without errors[0m
[38;5;243m 155[0m [38;5;249m  - [ ] No broken links[0m
[38;5;243m 156[0m [38;5;249m  - [ ] All images load correctly[0m
[38;5;243m 157[0m [38;5;249m  - [ ] Code examples render properly[0m
[38;5;243m 158[0m [38;5;249m- [ ] **Notebook Execution**: Run full notebook end-to-end[0m
[38;5;243m 159[0m [38;5;249m  - [ ] All cells execute without errors[0m
[38;5;243m 160[0m [38;5;249m  - [ ] Visualizations render correctly[0m
[38;5;243m 161[0m [38;5;249m  - [ ] No performance issues (reasonable execution time)[0m
[38;5;243m 162[0m [38;5;249m- [ ] **Code Review**: Request code review from team[0m
[38;5;243m 163[0m [38;5;249m- [ ] **Integration Test**: Test model in real-world scenario[0m
[38;5;243m 164[0m [38;5;249m- [ ] **Performance Test**: Verify model meets performance requirements[0m
[38;5;243m 165[0m 
[38;5;243m 166[0m [38;5;249m---[0m
[38;5;243m 167[0m 
[38;5;243m 168[0m [38;5;249m## Key Requirements[0m
[38;5;243m 169[0m 
[38;5;243m 170[0m [38;5;249m### ✅ Keras 3 Only[0m
[38;5;243m 171[0m [38;5;249mAll model implementations MUST use only Keras 3 operations and layers. NO TensorFlow dependencies are allowed in model implementations.[0m
[38;5;243m 172[0m [38;5;249m- **Allowed**: `keras.layers`, `keras.ops`, `kerasfactory.layers`, `kerasfactory.models`[0m
[38;5;243m 173[0m [38;5;249m- **NOT Allowed**: `tensorflow.python.*`, `tf.nn.*` (use `keras.ops.*` instead)[0m
[38;5;243m 174[0m [38;5;249m- **Exception**: TensorFlow can ONLY be used in test files and notebooks for validation[0m
[38;5;243m 175[0m 
[38;5;243m 176[0m [38;5;249m### ✅ Reusable Components[0m
[38;5;243m 177[0m [38;5;249mAvoid embedding layer logic directly in models. Create standalone, reusable layers first:[0m
[38;5;243m 178[0m [38;5;249m- **Good**: Implement `TemporalMixing` as a layer, use it in `TSMixer` model[0m
[38;5;243m 179[0m [38;5;249m- **Bad**: Implement temporal mixing logic directly in model[0m
[38;5;243m 180[0m 
[38;5;243m 181[0m [38;5;249m### ✅ Proper Inheritance[0m
[38;5;243m 182[0m [38;5;249m- Models must inherit from `kerasfactory.models._base.BaseModel`[0m
[38;5;243m 183[0m [38;5;249m- Layers must inherit from `kerasfactory.layers._base_layer.BaseLayer`[0m
[38;5;243m 184[0m 
[38;5;243m 185[0m [38;5;249m### ✅ Type Annotations (Python 3.12+)[0m
[38;5;243m 186[0m [38;5;249mUse modern type hints with the union operator:[0m
[38;5;243m 187[0m [38;5;249m```python[0m
[38;5;243m 188[0m [38;5;249mparam: int | float = 0.1  # Instead of Union[int, float][0m
[38;5;243m 189[0m [38;5;249m```[0m
[38;5;243m 190[0m 
[38;5;243m 191[0m [38;5;249m### ✅ Comprehensive Documentation[0m
[38;5;243m 192[0m [38;5;249mEvery model needs extensive documentation covering usage, architecture, and best practices.[0m
[38;5;243m 193[0m 
[38;5;243m 194[0m [38;5;249m---[0m
[38;5;243m 195[0m 
[38;5;243m 196[0m [38;5;249m## Implementation Pattern[0m
[38;5;243m 197[0m 
[38;5;243m 198[0m [38;5;249mFollow this pattern for implementing models:[0m
[38;5;243m 199[0m 
[38;5;243m 200[0m [38;5;249m```python[0m
[38;5;243m 201[0m [38;5;249m"""[0m
[38;5;243m 202[0m [38;5;249mModule docstring describing the model's purpose and functionality.[0m
[38;5;243m 203[0m [38;5;249m"""[0m
[38;5;243m 204[0m 
[38;5;243m 205[0m [38;5;249mfrom typing import Any[0m
[38;5;243m 206[0m [38;5;249mfrom loguru import logger[0m
[38;5;243m 207[0m [38;5;249mfrom keras import layers, ops[0m
[38;5;243m 208[0m [38;5;249mfrom keras import KerasTensor[0m
[38;5;243m 209[0m [38;5;249mfrom keras.saving import register_keras_serializable[0m
[38;5;243m 210[0m [38;5;249mfrom kerasfactory.models._base import BaseModel[0m
[38;5;243m 211[0m [38;5;249mfrom kerasfactory.layers import YourCustomLayer  # Use existing layers[0m
[38;5;243m 212[0m 
[38;5;243m 213[0m [38;5;249m@register_keras_serializable(package="kerasfactory.models")[0m
[38;5;243m 214[0m [38;5;249mclass YourCustomModel(BaseModel):[0m
[38;5;243m 215[0m [38;5;249m    """Comprehensive model description.[0m
[38;5;243m 216[0m [38;5;249m    [0m
[38;5;243m 217[0m [38;5;249m    This model implements [algorithm/architecture] for [task].[0m
[38;5;243m 218[0m [38;5;249m    It combines multiple layers to [describe what it does].[0m
[38;5;243m 219[0m [38;5;249m    [0m
[38;5;243m 220[0m [38;5;249m    Args:[0m
[38;5;243m 221[0m [38;5;249m        param1: Description with type and default.[0m
[38;5;243m 222[0m [38;5;249m        param2: Description with type and default.[0m
[38;5;243m 223[0m [38;5;249m        name: Optional name for the model.[0m
[38;5;243m 224[0m [38;5;249m    [0m
[38;5;243m 225[0m [38;5;249m    Input shape:[0m
[38;5;243m 226[0m [38;5;249m        `(batch_size, ...)` - Description of input.[0m
[38;5;243m 227[0m [38;5;249m    [0m
[38;5;243m 228[0m [38;5;249m    Output shape:[0m
[38;5;243m 229[0m [38;5;249m        `(batch_size, ...)` - Description of output.[0m
[38;5;243m 230[0m [38;5;249m    [0m
[38;5;243m 231[0m [38;5;249m    Example:[0m
[38;5;243m 232[0m [38;5;249m        ```python[0m
[38;5;243m 233[0m [38;5;249m        import keras[0m
[38;5;243m 234[0m [38;5;249m        from kerasfactory.models import YourCustomModel[0m
[38;5;243m 235[0m [38;5;249m        [0m
[38;5;243m 236[0m [38;5;249m        # Create model[0m
[38;5;243m 237[0m [38;5;249m        model = YourCustomModel(param1=value1, param2=value2)[0m
[38;5;243m 238[0m [38;5;249m        model.compile(optimizer='adam', loss='mse')[0m
[38;5;243m 239[0m [38;5;249m        [0m
[38;5;243m 240[0m [38;5;249m        # Train[0m
[38;5;243m 241[0m [38;5;249m        model.fit(X_train, y_train, epochs=10)[0m
[38;5;243m 242[0m [38;5;249m        [0m
[38;5;243m 243[0m [38;5;249m        # Predict[0m
[38;5;243m 244[0m [38;5;249m        predictions = model.predict(X_test)[0m
[38;5;243m 245[0m [38;5;249m        ```[0m
[38;5;243m 246[0m [38;5;249m    [0m
[38;5;243m 247[0m [38;5;249m    References:[0m
[38;5;243m 248[0m [38;5;249m        - Author et al. (Year). "Paper Title". Journal.[0m
[38;5;243m 249[0m [38;5;249m    """[0m
[38;5;243m 250[0m 
[38;5;243m 251[0m [38;5;249m    def __init__([0m
[38;5;243m 252[0m [38;5;249m        self,[0m
[38;5;243m 253[0m [38;5;249m        param1: int = 32,[0m
[38;5;243m 254[0m [38;5;249m        param2: float = 0.1,[0m
[38;5;243m 255[0m [38;5;249m        name: str | None = None,[0m
[38;5;243m 256[0m [38;5;249m        **kwargs: Any[0m
[38;5;243m 257[0m [38;5;249m    ) -> None:[0m
[38;5;243m 258[0m [38;5;249m        # Set private attributes[0m
[38;5;243m 259[0m [38;5;249m        self._param1 = param1[0m
[38;5;243m 260[0m [38;5;249m        self._param2 = param2[0m
[38;5;243m 261[0m 
[38;5;243m 262[0m [38;5;249m        # Validate parameters[0m
[38;5;243m 263[0m [38;5;249m        self._validate_params()[0m
[38;5;243m 264[0m 
[38;5;243m 265[0m [38;5;249m        # Set public attributes BEFORE super().__init__()[0m
[38;5;243m 266[0m [38;5;249m        self.param1 = self._param1[0m
[38;5;243m 267[0m [38;5;249m        self.param2 = self._param2[0m
[38;5;243m 268[0m 
[38;5;243m 269[0m [38;5;249m        # Call parent's __init__[0m
[38;5;243m 270[0m [38;5;249m        super().__init__(name=name, **kwargs)[0m
[38;5;243m 271[0m 
[38;5;243m 272[0m [38;5;249m    def _validate_params(self) -> None:[0m
[38;5;243m 273[0m [38;5;249m        """Validate model parameters."""[0m
[38;5;243m 274[0m [38;5;249m        if self._param1 < 1:[0m
[38;5;243m 275[0m [38;5;249m            raise ValueError(f"param1 must be >= 1, got {self._param1}")[0m
[38;5;243m 276[0m [38;5;249m        if not (0 <= self._param2 <= 1):[0m
[38;5;243m 277[0m [38;5;249m            raise ValueError(f"param2 must be in [0, 1], got {self._param2}")[0m
[38;5;243m 278[0m 
[38;5;243m 279[0m [38;5;249m    def build(self, input_shape: tuple[int, ...] | list[tuple[int, ...]]) -> None:[0m
[38;5;243m 280[0m [38;5;249m        """Build model with given input shape(s).[0m
[38;5;243m 281[0m 
[38;5;243m 282[0m [38;5;249m        Args:[0m
[38;5;243m 283[0m [38;5;249m            input_shape: Tuple(s) of integers defining input shape(s).[0m
[38;5;243m 284[0m [38;5;249m        """[0m
[38;5;243m 285[0m [38;5;249m        # Initialize all layers[0m
[38;5;243m 286[0m [38;5;249m        self.layer1 = YourCustomLayer(self._param1)[0m
[38;5;243m 287[0m [38;5;249m        self.layer2 = layers.Dense(self._param1)[0m
[38;5;243m 288[0m [38;5;249m        self.output_layer = layers.Dense(10)  # or task-specific output[0m
[38;5;243m 289[0m [38;5;249m        [0m
[38;5;243m 290[0m [38;5;249m        logger.debug(f"Building {self.__class__.__name__} with params: "[0m
[38;5;243m 291[0m [38;5;249m                    f"param1={self.param1}, param2={self.param2}")[0m
[38;5;243m 292[0m [38;5;249m        super().build(input_shape)[0m
[38;5;243m 293[0m 
[38;5;243m 294[0m [38;5;249m    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:[0m
[38;5;243m 295[0m [38;5;249m        """Forward pass.[0m
[38;5;243m 296[0m 
[38;5;243m 297[0m [38;5;249m        Args:[0m
[38;5;243m 298[0m [38;5;249m            inputs: Input tensor(s).[0m
[38;5;243m 299[0m [38;5;249m            training: Whether in training mode.[0m
[38;5;243m 300[0m 
[38;5;243m 301[0m [38;5;249m        Returns:[0m
[38;5;243m 302[0m [38;5;249m            Model output tensor.[0m
[38;5;243m 303[0m [38;5;249m        """[0m
[38;5;243m 304[0m [38;5;249m        # Forward pass through layers[0m
[38;5;243m 305[0m [38;5;249m        x = self.layer1(inputs, training=training)[0m
[38;5;243m 306[0m [38;5;249m        x = self.layer2(x)[0m
[38;5;243m 307[0m [38;5;249m        output = self.output_layer(x)[0m
[38;5;243m 308[0m [38;5;249m        return output[0m
[38;5;243m 309[0m 
[38;5;243m 310[0m [38;5;249m    def get_config(self) -> dict[str, Any]:[0m
[38;5;243m 311[0m [38;5;249m        """Returns model configuration.[0m
[38;5;243m 312[0m 
[38;5;243m 313[0m [38;5;249m        Returns:[0m
[38;5;243m 314[0m [38;5;249m            Dictionary with model configuration.[0m
[38;5;243m 315[0m [38;5;249m        """[0m
[38;5;243m 316[0m [38;5;249m        config = super().get_config()[0m
[38;5;243m 317[0m [38;5;249m        config.update({[0m
[38;5;243m 318[0m [38;5;249m            "param1": self.param1,[0m
[38;5;243m 319[0m [38;5;249m            "param2": self.param2,[0m
[38;5;243m 320[0m [38;5;249m        })[0m
[38;5;243m 321[0m [38;5;249m        return config[0m
[38;5;243m 322[0m [38;5;249m```[0m
[38;5;243m 323[0m 
[38;5;243m 324[0m [38;5;249m---[0m
[38;5;243m 325[0m 
[38;5;243m 326[0m [38;5;249m## Model Serialization & Loading[0m
[38;5;243m 327[0m 
[38;5;243m 328[0m [38;5;249mEnsure your model can be saved and loaded correctly:[0m
[38;5;243m 329[0m 
[38;5;243m 330[0m [38;5;249m```python[0m
[38;5;243m 331[0m [38;5;249mimport keras[0m
[38;5;243m 332[0m [38;5;249mimport tempfile[0m
[38;5;243m 333[0m 
[38;5;243m 334[0m [38;5;249m# Create and train model[0m
[38;5;243m 335[0m [38;5;249mmodel = YourCustomModel(param1=32, param2=0.1)[0m
[38;5;243m 336[0m [38;5;249mmodel.compile(optimizer='adam', loss='mse', metrics=['mae'])[0m
[38;5;243m 337[0m [38;5;249mmodel.fit(X_train, y_train, epochs=10, verbose=0)[0m
[38;5;243m 338[0m 
[38;5;243m 339[0m [38;5;249m# Save full model[0m
[38;5;243m 340[0m [38;5;249mwith tempfile.TemporaryDirectory() as tmpdir:[0m
[38;5;243m 341[0m [38;5;249m    # Save with architecture[0m
[38;5;243m 342[0m [38;5;249m    model.save(f'{tmpdir}/model.keras')[0m
[38;5;243m 343[0m [38;5;249m    [0m
[38;5;243m 344[0m [38;5;249m    # Load full model[0m
[38;5;243m 345[0m [38;5;249m    loaded_model = keras.models.load_model(f'{tmpdir}/model.keras')[0m
[38;5;243m 346[0m [38;5;249m    [0m
[38;5;243m 347[0m [38;5;249m    # Verify predictions are identical[0m
[38;5;243m 348[0m [38;5;249m    pred1 = model.predict(X_test)[0m
[38;5;243m 349[0m [38;5;249m    pred2 = loaded_model.predict(X_test)[0m
[38;5;243m 350[0m [38;5;249m    [0m
[38;5;243m 351[0m [38;5;249m    # Save only weights[0m
[38;5;243m 352[0m [38;5;249m    model.save_weights(f'{tmpdir}/weights.h5')[0m
[38;5;243m 353[0m [38;5;249m    [0m
[38;5;243m 354[0m [38;5;249m    # Load weights into new model[0m
[38;5;243m 355[0m [38;5;249m    new_model = YourCustomModel(param1=32, param2=0.1)[0m
[38;5;243m 356[0m [38;5;249m    new_model.load_weights(f'{tmpdir}/weights.h5')[0m
[38;5;243m 357[0m [38;5;249m```[0m
[38;5;243m 358[0m 
[38;5;243m 359[0m [38;5;249m---[0m
[38;5;243m 360[0m 
[38;5;243m 361[0m [38;5;249m## Testing Template[0m
[38;5;243m 362[0m 
[38;5;243m 363[0m [38;5;249mCreate comprehensive tests following this template:[0m
[38;5;243m 364[0m 
[38;5;243m 365[0m [38;5;249m```python[0m
[38;5;243m 366[0m [38;5;249mimport unittest[0m
[38;5;243m 367[0m [38;5;249mimport numpy as np[0m
[38;5;243m 368[0m [38;5;249mimport tensorflow as tf[0m
[38;5;243m 369[0m [38;5;249mimport keras[0m
[38;5;243m 370[0m 
[38;5;243m 371[0m [38;5;249mfrom kerasfactory.models import YourCustomModel[0m
[38;5;243m 372[0m 
[38;5;243m 373[0m [38;5;249mclass TestYourCustomModel(unittest.TestCase):[0m
[38;5;243m 374[0m [38;5;249m    """Test suite for YourCustomModel."""[0m
[38;5;243m 375[0m 
[38;5;243m 376[0m [38;5;249m    def setUp(self) -> None:[0m
[38;5;243m 377[0m [38;5;249m        """Set up test fixtures."""[0m
[38;5;243m 378[0m [38;5;249m        self.model = YourCustomModel(param1=32, param2=0.1)[0m
[38;5;243m 379[0m [38;5;249m        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])[0m
[38;5;243m 380[0m [38;5;249m        [0m
[38;5;243m 381[0m [38;5;249m        # Create sample data[0m
[38;5;243m 382[0m [38;5;249m        self.X_train = np.random.randn(100, 20).astype(np.float32)[0m
[38;5;243m 383[0m [38;5;249m        self.y_train = np.random.randn(100, 10).astype(np.float32)[0m
[38;5;243m 384[0m [38;5;249m        self.X_test = np.random.randn(20, 20).astype(np.float32)[0m
[38;5;243m 385[0m [38;5;249m        self.y_test = np.random.randn(20, 10).astype(np.float32)[0m
[38;5;243m 386[0m 
[38;5;243m 387[0m [38;5;249m    def test_initialization(self) -> None:[0m
[38;5;243m 388[0m [38;5;249m        """Test model initialization."""[0m
[38;5;243m 389[0m [38;5;249m        self.assertEqual(self.model.param1, 32)[0m
[38;5;243m 390[0m [38;5;249m        self.assertEqual(self.model.param2, 0.1)[0m
[38;5;243m 391[0m 
[38;5;243m 392[0m [38;5;249m    def test_invalid_parameters(self) -> None:[0m
[38;5;243m 393[0m [38;5;249m        """Test invalid parameter handling."""[0m
[38;5;243m 394[0m [38;5;249m        with self.assertRaises(ValueError):[0m
[38;5;243m 395[0m [38;5;249m            YourCustomModel(param1=-1)[0m
[38;5;243m 396[0m 
[38;5;243m 397[0m [38;5;249m    def test_forward_pass(self) -> None:[0m
[38;5;243m 398[0m [38;5;249m        """Test forward pass."""[0m
[38;5;243m 399[0m [38;5;249m        output = self.model(self.X_test)[0m
[38;5;243m 400[0m [38;5;249m        self.assertEqual(output.shape, (20, 10))[0m
[38;5;243m 401[0m 
[38;5;243m 402[0m [38;5;249m    def test_training(self) -> None:[0m
[38;5;243m 403[0m [38;5;249m        """Test model training."""[0m
[38;5;243m 404[0m [38;5;249m        history = self.model.fit([0m
[38;5;243m 405[0m [38;5;249m            self.X_train, self.y_train,[0m
[38;5;243m 406[0m [38;5;249m            epochs=2, batch_size=32, verbose=0[0m
[38;5;243m 407[0m [38;5;249m        )[0m
[38;5;243m 408[0m [38;5;249m        [0m
[38;5;243m 409[0m [38;5;249m        # Verify training occurred (loss changed)[0m
[38;5;243m 410[0m [38;5;249m        self.assertIsNotNone(history.history['loss'])[0m
[38;5;243m 411[0m 
[38;5;243m 412[0m [38;5;249m    def test_serialization(self) -> None:[0m
[38;5;243m 413[0m [38;5;249m        """Test model serialization."""[0m
[38;5;243m 414[0m [38;5;249m        config = self.model.get_config()[0m
[38;5;243m 415[0m [38;5;249m        new_model = YourCustomModel.from_config(config)[0m
[38;5;243m 416[0m [38;5;249m        new_model.compile(optimizer='adam', loss='mse')[0m
[38;5;243m 417[0m [38;5;249m        [0m
[38;5;243m 418[0m [38;5;249m        output1 = self.model(self.X_test)[0m
[38;5;243m 419[0m [38;5;249m        output2 = new_model(self.X_test)[0m
[38;5;243m 420[0m [38;5;249m        [0m
[38;5;243m 421[0m [38;5;249m        np.testing.assert_allclose(output1, output2, rtol=1e-5)[0m
[38;5;243m 422[0m 
[38;5;243m 423[0m [38;5;249m    def test_save_load(self) -> None:[0m
[38;5;243m 424[0m [38;5;249m        """Test model save and load."""[0m
[38;5;243m 425[0m [38;5;249m        import tempfile[0m
[38;5;243m 426[0m [38;5;249m        [0m
[38;5;243m 427[0m [38;5;249m        with tempfile.TemporaryDirectory() as tmpdir:[0m
[38;5;243m 428[0m [38;5;249m            model_path = f'{tmpdir}/model.keras'[0m
[38;5;243m 429[0m [38;5;249m            self.model.save(model_path)[0m
[38;5;243m 430[0m [38;5;249m            [0m
[38;5;243m 431[0m [38;5;249m            loaded_model = keras.models.load_model(model_path)[0m
[38;5;243m 432[0m [38;5;249m            loaded_model.compile(optimizer='adam', loss='mse')[0m
[38;5;243m 433[0m [38;5;249m            [0m
[38;5;243m 434[0m [38;5;249m            pred1 = self.model.predict(self.X_test, verbose=0)[0m
[38;5;243m 435[0m [38;5;249m            pred2 = loaded_model.predict(self.X_test, verbose=0)[0m
[38;5;243m 436[0m [38;5;249m            [0m
[38;5;243m 437[0m [38;5;249m            np.testing.assert_allclose(pred1, pred2, rtol=1e-5)[0m
[38;5;243m 438[0m 
[38;5;243m 439[0m [38;5;249mif __name__ == "__main__":[0m
[38;5;243m 440[0m [38;5;249m    unittest.main()[0m
[38;5;243m 441[0m [38;5;249m```[0m
[38;5;243m 442[0m 
[38;5;243m 443[0m [38;5;249m---[0m
[38;5;243m 444[0m 
[38;5;243m 445[0m [38;5;249m## Common Pitfalls & Solutions[0m
[38;5;243m 446[0m 
[38;5;243m 447[0m [38;5;249m| Pitfall | Problem | Solution |[0m
[38;5;243m 448[0m [38;5;249m|---------|---------|----------|[0m
[38;5;243m 449[0m [38;5;249m| Embedded layer logic | Code not reusable | Create standalone layers first |[0m
[38;5;243m 450[0m [38;5;249m| TensorFlow dependencies | Using `tf.*` operations | Use `keras.ops.*` and `kerasfactory.layers` |[0m
[38;5;243m 451[0m [38;5;249m| Wrong inheritance | Type errors | Inherit from `BaseModel` |[0m
[38;5;243m 452[0m [38;5;249m| Incomplete serialization | Cannot save/load | Include all parameters in `get_config()` |[0m
[38;5;243m 453[0m [38;5;249m| Missing layer instantiation | Runtime errors | Initialize all layers in `build()` |[0m
[38;5;243m 454[0m [38;5;249m| Wrong attribute order | `AttributeError` | Set public attributes BEFORE `super().__init__()` |[0m
[38;5;243m 455[0m [38;5;249m| Insufficient tests | Bugs in production | Write comprehensive tests |[0m
[38;5;243m 456[0m [38;5;249m| Inadequate documentation | Users confused | Write detailed guide with examples |[0m
[38;5;243m 457[0m [38;5;249m| No notebook example | Hard to get started | Create end-to-end demo notebook |[0m
[38;5;243m 458[0m 
[38;5;243m 459[0m [38;5;249m---[0m
[38;5;243m 460[0m 
[38;5;243m 461[0m [38;5;249m## Next Steps[0m
[38;5;243m 462[0m 
[38;5;243m 463[0m [38;5;249mAfter implementing and testing your model:[0m
[38;5;243m 464[0m 
[38;5;243m 465[0m [38;5;249m1. **Submit for Review**: Create a pull request with your implementation[0m
[38;5;243m 466[0m [38;5;249m2. **Address Feedback**: Update based on review comments[0m
[38;5;243m 467[0m [38;5;249m3. **Final Testing**: Run full test suite one more time[0m
[38;5;243m 468[0m [38;5;249m4. **Merge**: Once approved, merge to main branch[0m
[38;5;243m 469[0m [38;5;249m5. **Announce**: Notify team about new model availability[0m
[38;5;243m 470[0m [38;5;249m6. **Update README**: Add to main README and features list[0m
[38;5;243m 471[0m 
[38;5;243m 472[0m [38;5;249m---[0m
[38;5;243m 473[0m 
[38;5;243m 474[0m [38;5;249m## Related Resources[0m
[38;5;243m 475[0m 
[38;5;243m 476[0m [38;5;249m- [Layer Implementation Guide](layers_implementation_guide.md) - Detailed layer implementation guide[0m
[38;5;243m 477[0m [38;5;249m- [API Reference - Models](api/models.md) - Model API documentation[0m
[38;5;243m 478[0m [38;5;249m- [Contributing Guidelines](contributing.md) - Project contribution guidelines[0m
[38;5;243m 479[0m [38;5;249m- [Keras 3 Documentation](https://keras.io/api/) - Keras 3 API reference[0m
[38;5;243m 480[0m 
