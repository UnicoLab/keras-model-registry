---
title: Layers Explorer
description: Interactive explorer for all KMR layers with search, filtering, and detailed information
keywords: [layers, explorer, search, filter, keras, tabular, attention]
---

<style>
.explorer-intro {
  background: linear-gradient(135deg, var(--md-primary-fg-color) 0%, var(--md-primary-fg-color-light) 100%);
  color: white;
  padding: 3rem 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  text-align: center;
}

.explorer-intro h1 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  font-weight: 700;
}

.explorer-intro-subtitle {
  font-size: 1.1rem;
  margin-bottom: 2rem;
  opacity: 0.95;
  line-height: 1.6;
}

.explorer-stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.2);
}

.stat-item {
  text-align: center;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  display: block;
  margin-bottom: 0.5rem;
}

.stat-label {
  font-size: 0.9rem;
  opacity: 0.9;
}

.search-section {
  background: var(--md-default-bg-color);
  padding: 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  box-shadow: var(--md-shadow-z1);
}

.search-title {
  font-size: 1.3rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: var(--md-primary-fg-color);
}

.filter-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}

.filter-group-modern {
  display: flex;
  flex-direction: column;
}

.filter-group-modern label {
  font-weight: 600;
  margin-bottom: 0.75rem;
  color: var(--md-default-fg-color-70);
  font-size: 0.9rem;
}

.filter-buttons-modern {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.filter-btn-modern {
  padding: 0.5rem 1rem;
  border: 1px solid var(--md-default-fg-color-20);
  border-radius: 6px;
  background: var(--md-default-bg-color);
  color: var(--md-default-fg-color);
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.filter-btn-modern:hover {
  border-color: var(--md-primary-fg-color);
  color: var(--md-primary-fg-color);
  background: var(--md-primary-fg-color-light);
}

.filter-btn-modern.active {
  border-color: var(--md-primary-fg-color);
  background: var(--md-primary-fg-color);
  color: white;
  font-weight: 600;
}

.results-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  padding: 1rem;
  background: var(--md-default-bg-color);
  border-radius: 8px;
}

.results-count {
  font-weight: 600;
  color: var(--md-primary-fg-color);
}

.view-toggle-modern {
  display: flex;
  gap: 0.5rem;
}

.view-btn-modern {
  padding: 0.5rem 1rem;
  border: 1px solid var(--md-default-fg-color-20);
  border-radius: 6px;
  background: var(--md-default-bg-color);
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.3s ease;
}

.view-btn-modern.active {
  border-color: var(--md-primary-fg-color);
  background: var(--md-primary-fg-color);
  color: white;
}

@media (max-width: 768px) {
  .explorer-intro h1 {
    font-size: 2rem;
  }
  
  .explorer-stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .filter-row {
    grid-template-columns: 1fr;
  }
}
</style>

<div class="explorer-intro">
  <h1>🧩 Layers Explorer</h1>
  <div class="explorer-intro-subtitle">
    Discover and explore <strong>38+ powerful layers</strong> in the Keras Model Registry. <br>
    Find the perfect layer for your tabular modeling needs with our interactive explorer.
  </div>
  
  <div class="explorer-stats-grid">
    <div class="stat-item">
      <span class="stat-value">38+</span>
      <span class="stat-label">Powerful Layers</span>
    </div>
    <div class="stat-item">
      <span class="stat-value">5</span>
      <span class="stat-label">Categories</span>
    </div>
    <div class="stat-item">
      <span class="stat-value">100%</span>
      <span class="stat-label">Keras 3 Native</span>
    </div>
    <div class="stat-item">
      <span class="stat-value">0%</span>
      <span class="stat-label">TensorFlow Lock-in</span>
    </div>
  </div>
</div>

<div class="search-section">
  <div class="search-title">🔍 Smart Search & Advanced Filtering</div>
  
  <div class="search-box">
    <input type="search" placeholder="🔎 Search by name, description, or use case..." id="layer-search" style="width: 100%; padding: 0.75rem 1rem; border: 1px solid var(--md-default-fg-color-20); border-radius: 8px; font-size: 1rem;">
    <button class="search-clear" onclick="clearSearch()" style="float: right; margin-top: -2.5rem; margin-right: 0.5rem; background: none; border: none; cursor: pointer; font-size: 1.2rem;">✕</button>
  </div>

  <div class="filter-row" style="margin-top: 1.5rem;">
    <div class="filter-group-modern">
      <label>📁 Category</label>
      <div class="filter-buttons-modern">
        <button class="filter-btn-modern active" data-category="all">All (38)</button>
        <button class="filter-btn-modern" data-category="attention">🧠 Attention</button>
        <button class="filter-btn-modern" data-category="preprocessing">🔧 Preprocessing</button>
        <button class="filter-btn-modern" data-category="feature-engineering">⚙️ Feature Eng.</button>
        <button class="filter-btn-modern" data-category="specialized">🏗️ Specialized</button>
        <button class="filter-btn-modern" data-category="utility">🛠️ Utility</button>
      </div>
    </div>

    <div class="filter-group-modern">
      <label>📊 Complexity</label>
      <div class="filter-buttons-modern">
        <button class="filter-btn-modern active" data-complexity="all">All</button>
        <button class="filter-btn-modern" data-complexity="beginner">🟢 Beginner</button>
        <button class="filter-btn-modern" data-complexity="intermediate">🟡 Intermediate</button>
        <button class="filter-btn-modern" data-complexity="advanced">🔴 Advanced</button>
      </div>
    </div>

    <div class="filter-group-modern">
      <label>⚡ Performance</label>
      <div class="filter-buttons-modern">
        <button class="filter-btn-modern active" data-performance="all">All</button>
        <button class="filter-btn-modern" data-performance="fast">⚡ Fast</button>
        <button class="filter-btn-modern" data-performance="memory">💾 Memory Eff.</button>
        <button class="filter-btn-modern" data-performance="accurate">🎯 Accurate</button>
      </div>
    </div>
  </div>
</div>

<div class="results-info">
  <span class="results-count" id="result-count">Showing all 38+ layers</span>
  <div class="view-toggle-modern">
    <button class="view-btn-modern active" data-view="grid">⊞ Grid View</button>
    <button class="view-btn-modern" data-view="list">☰ List View</button>
  </div>
</div>

---

## 🏆 Featured Layers

<div class="layers-grid" id="layers-grid">

<!-- Attention Layers -->
<div class="layer-card" data-category="attention" data-complexity="intermediate" data-use-case="tabular" data-performance="fast">
  <div class="layer-header">
    <h3>🧠 TabularAttention</h3>
    <div class="layer-badges">
      <span class="badge badge-popular">🔥 Popular</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-featured">⭐ Featured</span>
    </div>
  </div>
  
  <p class="layer-description">Dual attention mechanism for inter-feature and inter-sample relationships in tabular data. Automatically discovers complex feature interactions.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import TabularAttention

attention = TabularAttention(
    num_heads=8,
    key_dim=64,
    dropout=0.1
)
output = attention(inputs)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator fast">⚡ Fast</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
    <span class="perf-indicator accurate">🎯 High Accuracy</span>
  </div>

  <div class="layer-actions">
    <a href="layers/tabular-attention/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/tabular-attention/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<div class="layer-card" data-category="attention" data-complexity="advanced" data-use-case="tabular" data-performance="accurate">
  <div class="layer-header">
    <h3>🧠 MultiResolutionTabularAttention</h3>
    <div class="layer-badges">
      <span class="badge badge-advanced">🚀 Advanced</span>
      <span class="badge badge-stable">✅ Stable</span>
    </div>
  </div>
  
  <p class="layer-description">Multi-resolution attention mechanism that processes numerical and categorical features separately, then combines them with cross-attention.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import MultiResolutionTabularAttention

attention = MultiResolutionTabularAttention(
    num_heads=4,
    key_dim=32,
    dropout=0.1
)
output = attention(inputs)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator accurate">🎯 High Accuracy</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
  </div>

  <div class="layer-actions">
    <a href="layers/multi-resolution-tabular-attention/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/multi-resolution-tabular-attention/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<!-- Feature Engineering Layers -->
<div class="layer-card" data-category="feature-engineering" data-complexity="beginner" data-use-case="tabular" data-performance="fast">
  <div class="layer-header">
    <h3>⚙️ VariableSelection</h3>
    <div class="layer-badges">
      <span class="badge badge-popular">🔥 Popular</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-beginner">🟢 Beginner</span>
    </div>
  </div>
  
  <p class="layer-description">Intelligently selects the most relevant features for your model using learnable gating mechanisms and residual connections.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import VariableSelection

selector = VariableSelection(
    num_features=10,
    dropout=0.1
)
output = selector(inputs)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator fast">⚡ Fast</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
    <span class="perf-indicator accurate">🎯 High Accuracy</span>
  </div>

  <div class="layer-actions">
    <a href="layers/variable-selection/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/variable-selection/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<div class="layer-card" data-category="feature-engineering" data-complexity="intermediate" data-use-case="tabular" data-performance="accurate">
  <div class="layer-header">
    <h3>⚙️ GatedFeatureFusion</h3>
    <div class="layer-badges">
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-intermediate">🟡 Intermediate</span>
    </div>
  </div>
  
  <p class="layer-description">Combines two feature representations using a learned gating mechanism to balance their contributions optimally.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import GatedFeatureFusion

fusion = GatedFeatureFusion(
    hidden_dim=64,
    dropout=0.1
)
output = fusion([features1, features2])</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator accurate">🎯 High Accuracy</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
  </div>

  <div class="layer-actions">
    <a href="layers/gated-feature-fusion/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/gated-feature-fusion/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<!-- Preprocessing Layers -->
<div class="layer-card" data-category="preprocessing" data-complexity="intermediate" data-use-case="tabular" data-performance="fast">
  <div class="layer-header">
    <h3>🔧 DifferentiableTabularPreprocessor</h3>
    <div class="layer-badges">
      <span class="badge badge-popular">🔥 Popular</span>
      <span class="badge badge-stable">✅ Stable</span>
    </div>
  </div>
  
  <p class="layer-description">End-to-end differentiable preprocessing for tabular data with learnable imputation and normalization.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import DifferentiableTabularPreprocessor

preprocessor = DifferentiableTabularPreprocessor(
    numerical_features=['age', 'income'],
    categorical_features=['category', 'region']
)
output = preprocessor(inputs)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator fast">⚡ Fast</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
    <span class="perf-indicator accurate">🎯 High Accuracy</span>
  </div>

  <div class="layer-actions">
    <a href="layers/differentiable-tabular-preprocessor/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/differentiable-tabular-preprocessor/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<div class="layer-card" data-category="preprocessing" data-complexity="beginner" data-use-case="tabular" data-performance="fast">
  <div class="layer-header">
    <h3>🔧 DateEncodingLayer</h3>
    <div class="layer-badges">
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-beginner">🟢 Beginner</span>
    </div>
  </div>
  
  <p class="layer-description">Encodes date components into cyclical features using sine and cosine transformations for better temporal modeling.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import DateEncodingLayer

date_encoder = DateEncodingLayer(
    min_year=1900,
    max_year=2100
)
output = date_encoder(date_components)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator fast">⚡ Fast</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
  </div>

  <div class="layer-actions">
    <a href="layers/date-encoding-layer/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/date-encoding-layer/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<!-- Specialized Layers -->
<div class="layer-card" data-category="specialized" data-complexity="advanced" data-use-case="tabular" data-performance="accurate">
  <div class="layer-header">
    <h3>🏗️ GatedResidualNetwork</h3>
    <div class="layer-badges">
      <span class="badge badge-popular">🔥 Popular</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-advanced">🚀 Advanced</span>
    </div>
  </div>
  
  <p class="layer-description">Gated residual network for complex feature interactions with improved gradient flow and feature transformation.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import GatedResidualNetwork

grn = GatedResidualNetwork(
    units=128,
    dropout_rate=0.1
)
output = grn(inputs)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator accurate">🎯 High Accuracy</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
  </div>

  <div class="layer-actions">
    <a href="layers/gated-residual-network/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/gated-residual-network/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<div class="layer-card" data-category="specialized" data-complexity="advanced" data-use-case="tabular" data-performance="accurate">
  <div class="layer-header">
    <h3>🏗️ TabularMoELayer</h3>
    <div class="layer-badges">
      <span class="badge badge-advanced">🚀 Advanced</span>
      <span class="badge badge-stable">✅ Stable</span>
    </div>
  </div>
  
  <p class="layer-description">Mixture of Experts layer that routes input features through multiple expert networks with learnable gating.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import TabularMoELayer

moe = TabularMoELayer(
    num_experts=4,
    expert_units=16
)
output = moe(inputs)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator accurate">🎯 High Accuracy</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
  </div>

  <div class="layer-actions">
    <a href="layers/tabular-moe-layer/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/tabular-moe-layer/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<!-- Utility Layers -->
<div class="layer-card" data-category="utility" data-complexity="beginner" data-use-case="tabular" data-performance="fast">
  <div class="layer-header">
    <h3>🛠️ CastToFloat32Layer</h3>
    <div class="layer-badges">
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-beginner">🟢 Beginner</span>
    </div>
  </div>
  
  <p class="layer-description">Simple utility layer that casts input tensors to float32 data type for consistent data types in models.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import CastToFloat32Layer

cast_layer = CastToFloat32Layer()
output = cast_layer(inputs)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator fast">⚡ Fast</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
  </div>

  <div class="layer-actions">
    <a href="layers/cast-to-float32-layer/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/cast-to-float32-layer/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

<div class="layer-card" data-category="utility" data-complexity="intermediate" data-use-case="tabular" data-performance="accurate">
  <div class="layer-header">
    <h3>🛠️ NumericalAnomalyDetection</h3>
    <div class="layer-badges">
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-intermediate">🟡 Intermediate</span>
    </div>
  </div>
  
  <p class="layer-description">Detects numerical anomalies using statistical methods and learned thresholds for robust data processing.</p>

  <div class="code-snippet">
    <div class="code-header">
      <span>Quick Example</span>
      <button class="copy-btn" onclick="copyCode(this)">📋</button>
    </div>
    <pre><code class="language-python">from kmr.layers import NumericalAnomalyDetection

anomaly_detector = NumericalAnomalyDetection(
    threshold=2.0,
    method='zscore'
)
output = anomaly_detector(inputs)</code></pre>
  </div>

  <div class="layer-metadata">
    <span class="perf-indicator accurate">🎯 High Accuracy</span>
    <span class="perf-indicator memory">💾 Memory Efficient</span>
  </div>

  <div class="layer-actions">
    <a href="layers/numerical-anomaly-detection/" class="action-btn primary">📖 Full Docs</a>
    <a href="layers/numerical-anomaly-detection/#examples" class="action-btn secondary">💡 Examples</a>
  </div>
</div>

</div>

---

## 📚 Browse by Category

<div class="md-grid" markdown="1">

<div class="md-cell md-cell--4" markdown="1">
### 🧠 Attention Layers

!!! info "8 layers available"

**Core attention mechanisms for tabular data**

- TabularAttention
- MultiResolutionTabularAttention
- ColumnAttention
- RowAttention
- SparseAttentionWeighting

<a href="layers/tabular-attention/" class="md-button md-button--primary">Explore All →</a>
</div>

<div class="md-cell md-cell--4" markdown="1">
### 🔧 Preprocessing Layers

!!! info "6 layers available"

**Data preprocessing and transformation**

- DifferentiableTabularPreprocessor
- DifferentialPreprocessingLayer
- DateParsingLayer
- DateEncodingLayer
- CastToFloat32Layer
- SeasonLayer

<a href="layers/differentiable-tabular-preprocessor/" class="md-button md-button--primary">Explore All →</a>
</div>

<div class="md-cell md-cell--4" markdown="1">
### ⚙️ Feature Engineering

!!! info "8 layers available"

**Advanced feature engineering and selection**

- VariableSelection
- GatedFeatureSelection
- GatedFeatureFusion
- AdvancedNumericalEmbedding
- DistributionAwareEncoder
- DistributionTransformLayer

<a href="layers/variable-selection/" class="md-button md-button--primary">Explore All →</a>
</div>

<div class="md-cell md-cell--4" markdown="1">
### 🏗️ Specialized Layers

!!! info "10 layers available"

**Specialized architectures and techniques**

- GatedResidualNetwork
- GatedLinearUnit
- TransformerBlock
- TabularMoELayer
- BoostingBlock
- BoostingEnsembleLayer

<a href="layers/gated-residual-network/" class="md-button md-button--primary">Explore All →</a>
</div>

<div class="md-cell md-cell--4" markdown="1">
### 🛠️ Utility Layers

!!! info "6 layers available"

**Utility and helper layers**

- BusinessRulesLayer
- FeatureCutout
- StochasticDepth
- SlowNetwork
- HyperZZWOperator
- GraphFeatureAggregation

<a href="layers/advanced-graph-feature/" class="md-button md-button--primary">Explore All →</a>
</div>

</div>

---

## 📊 All Layers

<div class="md-grid" markdown="1">

<div class="md-cell md-cell--6" markdown="1">
### Attention Layers

!!! abstract "Core attention mechanisms for tabular data"

- [TabularAttention](layers/tabular-attention.md) - Dual attention for feature and sample relationships
- [MultiResolutionTabularAttention](layers/multi-resolution-tabular-attention.md) - Multi-resolution attention
- [ColumnAttention](layers/column-attention.md) - Column-wise attention mechanism
- [RowAttention](layers/row-attention.md) - Row-wise attention mechanism
- [SparseAttentionWeighting](layers/sparse-attention-weighting.md) - Sparse attention weights
</div>

<div class="md-cell md-cell--6" markdown="1">
### Preprocessing Layers

!!! abstract "Data preprocessing and transformation"

- [DifferentiableTabularPreprocessor](layers/differentiable-tabular-preprocessor.md) - End-to-end differentiable preprocessing
- [DifferentialPreprocessingLayer](layers/differential-preprocessing-layer.md) - Differential preprocessing
- [DateParsingLayer](layers/date-parsing-layer.md) - Date parsing and extraction
- [DateEncodingLayer](layers/date-encoding-layer.md) - Date feature encoding
- [CastToFloat32Layer](layers/cast-to-float32-layer.md) - Type casting utility
- [SeasonLayer](layers/season-layer.md) - Seasonal feature extraction
</div>

<div class="md-cell md-cell--6" markdown="1">
### Feature Engineering Layers

!!! abstract "Advanced feature engineering and selection"

- [VariableSelection](layers/variable-selection.md) - Intelligent feature selection
- [GatedFeatureSelection](layers/gated-feature-selection.md) - Gated feature selection
- [GatedFeatureFusion](layers/gated-feature-fusion.md) - Gated feature fusion
- [AdvancedNumericalEmbedding](layers/advanced-numerical-embedding.md) - Advanced numerical embeddings
- [DistributionAwareEncoder](layers/distribution-aware-encoder.md) - Distribution-aware encoding
- [DistributionTransformLayer](layers/distribution-transform-layer.md) - Distribution transformation
</div>

<div class="md-cell md-cell--6" markdown="1">
### Specialized Layers

!!! abstract "Specialized architectures and techniques"

- [GatedResidualNetwork](layers/gated-residual-network.md) - Gated residual network
- [GatedLinearUnit](layers/gated-linear-unit.md) - Gated linear unit
- [TransformerBlock](layers/transformer-block.md) - Transformer block for tabular data
- [TabularMoELayer](layers/tabular-moe-layer.md) - Mixture of Experts for tabular data
- [BoostingBlock](layers/boosting-block.md) - Boosting block implementation
- [BoostingEnsembleLayer](layers/boosting-ensemble-layer.md) - Boosting ensemble layer
</div>

<div class="md-cell md-cell--6" markdown="1">
### Utility Layers

!!! abstract "Utility and helper layers"

- [BusinessRulesLayer](layers/business-rules-layer.md) - Business rules enforcement
- [FeatureCutout](layers/feature-cutout.md) - Feature cutout augmentation
- [StochasticDepth](layers/stochastic-depth.md) - Stochastic depth regularization
- [SlowNetwork](layers/slow-network.md) - Multi-layer slow network
- [HyperZZWOperator](layers/hyper-zzw-operator.md) - HyperZZW operator
- [GraphFeatureAggregation](layers/graph-feature-aggregation.md) - Graph feature aggregation
</div>

</div>

---

## 🚀 Quick Start

!!! tip "Getting Started"

1. **Browse** the categories above to find relevant layers
2. **Search** for specific functionality using the search box
3. **Click** on any layer to view detailed documentation
4. **Copy** code examples to get started quickly

!!! example "Example Usage"

```python
import keras
from kmr.layers import TabularAttention, VariableSelection

# Create a simple model
inputs = keras.Input(shape=(10,))
x = VariableSelection(num_features=5)(inputs)
x = TabularAttention(num_heads=4, key_dim=32)(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)
```

---

## 📈 Performance Comparison

<div class="performance-comparison">
  <div class="comparison-header">
    <h3>Layer Performance Characteristics</h3>
    <p>Compare layers by their performance metrics and characteristics</p>
  </div>

  <div class="comparison-grid">
    <div class="comparison-card">
      <div class="comparison-category">
        <div class="category-icon">🧠</div>
        <h4>Attention Layers</h4>
        <span class="layer-count">5 layers</span>
      </div>
      <div class="performance-metrics">
        <div class="metric">
          <span class="metric-label">Speed</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 75%"></div>
          </div>
          <span class="metric-value">⚡⚡⚡</span>
        </div>
        <div class="metric">
          <span class="metric-label">Memory</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 60%"></div>
          </div>
          <span class="metric-value">💾💾</span>
        </div>
        <div class="metric">
          <span class="metric-label">Accuracy</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 90%"></div>
          </div>
          <span class="metric-value">🎯🎯🎯</span>
        </div>
      </div>
      <div class="use-cases">
        <span class="use-case-tag">Complex Interactions</span>
        <span class="use-case-tag">Feature Relationships</span>
        <span class="use-case-tag">Sample Relationships</span>
      </div>
    </div>

    <div class="comparison-card">
      <div class="comparison-category">
        <div class="category-icon">🔧</div>
        <h4>Preprocessing</h4>
        <span class="layer-count">6 layers</span>
      </div>
      <div class="performance-metrics">
        <div class="metric">
          <span class="metric-label">Speed</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 95%"></div>
          </div>
          <span class="metric-value">⚡⚡⚡⚡</span>
        </div>
        <div class="metric">
          <span class="metric-label">Memory</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 85%"></div>
          </div>
          <span class="metric-value">💾💾💾💾</span>
        </div>
        <div class="metric">
          <span class="metric-label">Accuracy</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 70%"></div>
          </div>
          <span class="metric-value">🎯🎯</span>
        </div>
      </div>
      <div class="use-cases">
        <span class="use-case-tag">Data Preparation</span>
        <span class="use-case-tag">Type Conversion</span>
        <span class="use-case-tag">Date Processing</span>
      </div>
    </div>

    <div class="comparison-card">
      <div class="comparison-category">
        <div class="category-icon">⚙️</div>
        <h4>Feature Engineering</h4>
        <span class="layer-count">8 layers</span>
      </div>
      <div class="performance-metrics">
        <div class="metric">
          <span class="metric-label">Speed</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 65%"></div>
          </div>
          <span class="metric-value">⚡⚡</span>
        </div>
        <div class="metric">
          <span class="metric-label">Memory</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 75%"></div>
          </div>
          <span class="metric-value">💾💾💾</span>
        </div>
        <div class="metric">
          <span class="metric-label">Accuracy</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 90%"></div>
          </div>
          <span class="metric-value">🎯🎯🎯</span>
        </div>
      </div>
      <div class="use-cases">
        <span class="use-case-tag">Feature Selection</span>
        <span class="use-case-tag">Feature Fusion</span>
        <span class="use-case-tag">Embeddings</span>
      </div>
    </div>

    <div class="comparison-card">
      <div class="comparison-category">
        <div class="category-icon">🏗️</div>
        <h4>Specialized</h4>
        <span class="layer-count">11 layers</span>
      </div>
      <div class="performance-metrics">
        <div class="metric">
          <span class="metric-label">Speed</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 60%"></div>
          </div>
          <span class="metric-value">⚡⚡</span>
        </div>
        <div class="metric">
          <span class="metric-label">Memory</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 70%"></div>
          </div>
          <span class="metric-value">💾💾💾</span>
        </div>
        <div class="metric">
          <span class="metric-label">Accuracy</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 95%"></div>
          </div>
          <span class="metric-value">🎯🎯🎯🎯</span>
        </div>
      </div>
      <div class="use-cases">
        <span class="use-case-tag">Advanced Architectures</span>
        <span class="use-case-tag">Mixture of Experts</span>
        <span class="use-case-tag">Boosting</span>
      </div>
    </div>

    <div class="comparison-card">
      <div class="comparison-category">
        <div class="category-icon">🛠️</div>
        <h4>Utility</h4>
        <span class="layer-count">8 layers</span>
      </div>
      <div class="performance-metrics">
        <div class="metric">
          <span class="metric-label">Speed</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 90%"></div>
          </div>
          <span class="metric-value">⚡⚡⚡⚡</span>
        </div>
        <div class="metric">
          <span class="metric-label">Memory</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 85%"></div>
          </div>
          <span class="metric-value">💾💾💾💾</span>
        </div>
        <div class="metric">
          <span class="metric-label">Accuracy</span>
          <div class="metric-bar">
            <div class="metric-fill" style="width: 60%"></div>
          </div>
          <span class="metric-value">🎯</span>
        </div>
      </div>
      <div class="use-cases">
        <span class="use-case-tag">Helper Functions</span>
        <span class="use-case-tag">Anomaly Detection</span>
        <span class="use-case-tag">Graph Processing</span>
      </div>
    </div>
  </div>
</div>

## 🎯 Layer Selection Guide

<div class="selection-guide">
  <div class="guide-section">
    <h3>🚀 Quick Start</h3>
    <p>New to KMR? Start with these beginner-friendly layers:</p>
    <div class="recommendation-cards">
      <div class="recommendation-card">
        <h4>VariableSelection</h4>
        <p>Automatic feature selection for any tabular dataset</p>
        <span class="difficulty beginner">🟢 Beginner</span>
      </div>
      <div class="recommendation-card">
        <h4>CastToFloat32Layer</h4>
        <p>Simple data type conversion utility</p>
        <span class="difficulty beginner">🟢 Beginner</span>
      </div>
      <div class="recommendation-card">
        <h4>DateEncodingLayer</h4>
        <p>Encode temporal features for better modeling</p>
        <span class="difficulty beginner">🟢 Beginner</span>
      </div>
    </div>
  </div>

  <div class="guide-section">
    <h3>🎯 Performance Optimization</h3>
    <p>Need speed and efficiency? These layers are optimized for performance:</p>
    <div class="recommendation-cards">
      <div class="recommendation-card">
        <h4>DifferentiableTabularPreprocessor</h4>
        <p>Fast, end-to-end preprocessing</p>
        <span class="performance fast">⚡ Fast</span>
      </div>
      <div class="recommendation-card">
        <h4>TabularAttention</h4>
        <p>Efficient attention for tabular data</p>
        <span class="performance fast">⚡ Fast</span>
      </div>
      <div class="recommendation-card">
        <h4>FeatureCutout</h4>
        <p>Lightweight data augmentation</p>
        <span class="performance fast">⚡ Fast</span>
      </div>
    </div>
  </div>

  <div class="guide-section">
    <h3>🏆 Maximum Accuracy</h3>
    <p>For the best possible results, use these high-accuracy layers:</p>
    <div class="recommendation-cards">
      <div class="recommendation-card">
        <h4>GatedResidualNetwork</h4>
        <p>Advanced feature interactions</p>
        <span class="performance accurate">🎯 High Accuracy</span>
      </div>
      <div class="recommendation-card">
        <h4>TabularMoELayer</h4>
        <p>Mixture of experts for complex patterns</p>
        <span class="performance accurate">🎯 High Accuracy</span>
      </div>
      <div class="recommendation-card">
        <h4>MultiResolutionTabularAttention</h4>
        <p>Multi-resolution attention mechanism</p>
        <span class="performance accurate">🎯 High Accuracy</span>
      </div>
    </div>
  </div>
</div>

---

## 🔗 Related Resources

- [API Reference](api/layers.md) - Complete API documentation
- [Tutorials](tutorials/basic-workflows.md) - Step-by-step guides
- [Examples](examples/README.md) - Real-world examples
- [Contributing](contributing.md) - How to contribute new layers

<script src="../assets/javascripts/layer-filtering.js"></script>
