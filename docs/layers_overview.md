---
title: 🧩 Layers - Complete Reference & Explorer
description: Complete reference for 36+ production-ready KerasFactory layers including attention mechanisms, feature engineering, preprocessing, and specialized architectures for tabular data with interactive explorer.
keywords: keras layers, tabular data layers, attention mechanisms, feature engineering, preprocessing layers, neural network layers, layer explorer
---

<style>
.layers-hero {
  background: linear-gradient(135deg, var(--md-primary-fg-color) 0%, var(--md-primary-fg-color-light) 100%);
  color: white;
  padding: 4rem 2rem;
  border-radius: 16px;
  margin-bottom: 3rem;
  text-align: center;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.layers-hero h1 {
  font-size: 3rem;
  margin-bottom: 1rem;
  font-weight: 700;
  letter-spacing: -1px;
}

.layers-tagline {
  font-size: 1.3rem;
  margin-bottom: 2rem;
  opacity: 0.95;
  line-height: 1.6;
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
}

.layers-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.2);
}

.stat-box {
  text-align: center;
}

.stat-number {
  font-size: 2.5rem;
  font-weight: 700;
  display: block;
  margin-bottom: 0.5rem;
}

.stat-text {
  font-size: 0.95rem;
  opacity: 0.9;
  font-weight: 500;
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

.filter-group {
  display: flex;
  flex-direction: column;
}

.filter-group label {
  font-weight: 600;
  margin-bottom: 0.75rem;
  color: var(--md-default-fg-color-70);
  font-size: 0.9rem;
}

.filter-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.filter-btn {
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

.filter-btn:hover {
  border-color: var(--md-primary-fg-color);
  color: var(--md-primary-fg-color);
  background: var(--md-primary-fg-color-light);
}

.filter-btn.active {
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

.view-toggle {
  display: flex;
  gap: 0.5rem;
}

.view-btn {
  padding: 0.5rem 1rem;
  border: 1px solid var(--md-default-fg-color-20);
  border-radius: 6px;
  background: var(--md-default-bg-color);
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.3s ease;
}

.view-btn.active {
  border-color: var(--md-primary-fg-color);
  background: var(--md-primary-fg-color);
  color: white;
}

.layers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
}

.layer-card {
  background: var(--md-default-bg-color);
  border-radius: 12px;
  padding: 1.5rem;
  border-left: 4px solid var(--md-primary-fg-color);
  box-shadow: var(--md-shadow-z1);
  transition: all 0.3s ease;
}

.layer-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--md-shadow-z2);
}

.layer-header {
  margin-bottom: 1rem;
}

.layer-header h3 {
  margin: 0 0 0.5rem 0;
  color: var(--md-primary-fg-color);
  font-size: 1.2rem;
}

.layer-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.badge {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
}

.badge-popular {
  background: #ff9800;
  color: white;
}

.badge-stable {
  background: #4caf50;
  color: white;
}

.badge-advanced {
  background: #2196f3;
  color: white;
}

.layer-description {
  font-size: 0.9rem;
  color: var(--md-default-fg-color-70);
  margin-bottom: 1rem;
  line-height: 1.5;
}

.layer-info {
  background: var(--md-default-bg-color-20);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  font-size: 0.85rem;
}

.layer-info-item {
  margin-bottom: 0.5rem;
}

.layer-info-item strong {
  color: var(--md-primary-fg-color);
}

.layer-actions {
  display: flex;
  gap: 0.75rem;
}

.action-btn {
  flex: 1;
  padding: 0.5rem;
  border-radius: 6px;
  text-align: center;
  text-decoration: none;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.2s ease;
  border: none;
  cursor: pointer;
}

.action-btn.primary {
  background: var(--md-primary-fg-color);
  color: white;
}

.action-btn.primary:hover {
  opacity: 0.9;
}

.action-btn.secondary {
  border: 1px solid var(--md-primary-fg-color);
  color: var(--md-primary-fg-color);
  background: transparent;
}

.action-btn.secondary:hover {
  background: var(--md-primary-fg-color-light);
}

.problem-solution-table {
  width: 100%;
  margin: 2rem 0;
  border-collapse: collapse;
  background: var(--md-default-bg-color);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: var(--md-shadow-z1);
}

.problem-solution-table th {
  background: var(--md-primary-fg-color-20);
  padding: 1rem;
  text-align: left;
  font-weight: 600;
  color: var(--md-primary-fg-color);
}

.problem-solution-table td {
  padding: 1.25rem;
  border-bottom: 1px solid var(--md-default-fg-color-10);
}

.problem-solution-table tr:last-child td {
  border-bottom: none;
}

.problem-col {
  color: #d32f2f;
  font-weight: 500;
}

.solution-col {
  color: #388e3c;
  font-weight: 500;
}

.highlight-box {
  background: var(--md-default-bg-color);
  border-left: 4px solid var(--md-primary-fg-color);
  padding: 1.5rem;
  border-radius: 8px;
  margin: 1.5rem 0;
}

.feature-highlights {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin: 2rem 0;
}

.feature-card {
  background: var(--md-default-bg-color);
  padding: 2rem;
  border-radius: 12px;
  box-shadow: var(--md-shadow-z1);
  text-align: center;
  border-top: 4px solid var(--md-primary-fg-color);
}

.feature-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.feature-card h3 {
  margin: 1rem 0;
  color: var(--md-primary-fg-color);
}

.feature-card p {
  font-size: 0.95rem;
  color: var(--md-default-fg-color-70);
  line-height: 1.6;
}

.api-reference-section {
  margin-top: 4rem;
}

.section-header {
  border-bottom: 2px solid var(--md-primary-fg-color);
  padding-bottom: 1rem;
  margin-bottom: 2rem;
}

.section-header h2 {
  margin: 0 0 0.5rem 0;
  color: var(--md-primary-fg-color);
}

.section-header p {
  color: var(--md-default-fg-color-70);
}

.api-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
  margin-bottom: 2rem;
}

.api-card {
  background: var(--md-default-bg-color);
  border-radius: 8px;
  padding: 1.5rem;
  border-left: 4px solid var(--md-primary-fg-color);
}

.api-card h4 {
  margin-top: 0;
  color: var(--md-primary-fg-color);
}

.api-signature {
  background: var(--md-default-bg-color-20);
  padding: 0.75rem;
  border-radius: 4px;
  font-family: monospace;
  font-size: 0.85rem;
  overflow-x: auto;
  margin-bottom: 1rem;
}

@media (max-width: 768px) {
  .layers-hero h1 {
    font-size: 2rem;
  }
  
  .layers-tagline {
    font-size: 1.1rem;
  }
  
  .layers-stats {
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
  }
  
  .filter-row {
    grid-template-columns: 1fr;
  }
  
  .layers-grid {
    grid-template-columns: 1fr;
  }
}
</style>

<div class="layers-hero">
  <h1>🧩 Layers - Complete Reference & Explorer</h1>
  <div class="layers-tagline">
    <strong>36+ production-ready layers</strong> designed exclusively for <strong>Keras 3</strong>.<br>
    Build sophisticated tabular models with advanced attention, feature engineering, and preprocessing layers.
  </div>
  
  <div class="layers-stats">
    <div class="stat-box">
      <span class="stat-number">36+</span>
      <span class="stat-text">Production Layers</span>
    </div>
    <div class="stat-box">
      <span class="stat-number">8</span>
      <span class="stat-text">Categories</span>
    </div>
    <div class="stat-box">
      <span class="stat-number">100%</span>
      <span class="stat-text">Keras 3 Native</span>
    </div>
    <div class="stat-box">
      <span class="stat-number">0%</span>
      <span class="stat-text">TensorFlow Lock-in</span>
    </div>
  </div>
</div>

## 🎯 Why Use KerasFactory Layers?

| Challenge | Traditional Approach | KerasFactory's Solution |
|-----------|----------------------|----------------|
| 🔗 **Feature Interactions** | Manual feature crosses | 👁️ **Tabular Attention** - Automatic relationship discovery |
| 🏷️ **Mixed Feature Types** | Uniform processing | 🧩 **Feature-wise Layers** - Specialized processing per feature |
| 📊 **Complex Distributions** | Fixed strategies | 📊 **Distribution-Aware Encoding** - Adaptive transformations |
| ⚡ **Performance Optimization** | Post-hoc analysis | 🎯 **Built-in Selection** - Learned during training |
| 🔒 **Production Readiness** | Extra tooling needed | ✅ **Battle-Tested** - Used in production models |

## ✨ Key Features

<div class="feature-highlights">
  <div class="feature-card">
    <div class="feature-icon">👁️</div>
    <h3>Attention Mechanisms</h3>
    <p>Automatically discover feature relationships and sample importance with advanced attention layers.</p>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🧩</div>
    <h3>Feature-wise Processing</h3>
    <p>Each feature receives specialized processing through mixture of experts and dedicated layers.</p>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">📊</div>
    <h3>Distribution-Aware</h3>
    <p>Automatically adapt to different distributions with intelligent encoding and transformations.</p>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">⚡</div>
    <h3>Performance Ready</h3>
    <p>Optimized for production with built-in regularization and efficient memory usage.</p>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🎯</div>
    <h3>Built-in Optimization</h3>
    <p>Learn which features matter during training, not after with integrated feature selection.</p>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🔒</div>
    <h3>Production Proven</h3>
    <p>Battle-tested in real-world ML pipelines with comprehensive testing and documentation.</p>
  </div>
</div>

<div class="highlight-box">
  <strong>💡 Pro Tip:</strong> Start with TabularAttention for feature relationships, VariableSelection for feature importance, and DifferentiableTabularPreprocessor for end-to-end preprocessing. Combine them for powerful custom architectures.
</div>

---

## 🔍 Interactive Layers Explorer

<div class="search-section">
  <div class="search-title">🔍 Smart Search & Advanced Filtering</div>
  
  <input type="search" placeholder="🔎 Search by name, description, or use case..." id="layer-search" style="width: 100%; padding: 0.75rem 1rem; border: 1px solid var(--md-default-fg-color-20); border-radius: 8px; font-size: 1rem; margin-bottom: 1.5rem;">

  <div class="filter-row">
    <div class="filter-group">
      <label>📁 Category</label>
      <div class="filter-buttons">
        <button class="filter-btn active" data-category="all">All (36)</button>
        <button class="filter-btn" data-category="attention">🧠 Attention</button>
        <button class="filter-btn" data-category="preprocessing">🔧 Preprocessing</button>
        <button class="filter-btn" data-category="feature-engineering">⚙️ Feature Eng.</button>
        <button class="filter-btn" data-category="ensemble">🏗️ Ensemble</button>
        <button class="filter-btn" data-category="utility">🛠️ Utility</button>
      </div>
    </div>

    <div class="filter-group">
      <label>📊 Complexity</label>
      <div class="filter-buttons">
        <button class="filter-btn active" data-complexity="all">All</button>
        <button class="filter-btn" data-complexity="beginner">🟢 Beginner</button>
        <button class="filter-btn" data-complexity="intermediate">🟡 Intermediate</button>
        <button class="filter-btn" data-complexity="advanced">🔴 Advanced</button>
      </div>
    </div>
  </div>
</div>

<div class="results-info">
  <span class="results-count" id="result-count">Showing all 36+ layers</span>
</div>

---

## 📚 All Layers by Category

### ⏱️ Time Series & Forecasting (16 layers)

**Specialized layers for time series forecasting, decomposition, and feature extraction with multi-scale pattern recognition.**

- **PositionalEmbedding** - Sinusoidal positional encoding for sequence models
- **FixedEmbedding** - Non-trainable embeddings for temporal indices (months, days, hours)
- **TokenEmbedding** - 1D convolution-based embedding for time series values
- **TemporalEmbedding** - Embedding layer for temporal features (month, day, weekday, hour, minute)
- **DataEmbeddingWithoutPosition** - Combined token and temporal embedding for comprehensive features
- **MovingAverage** - Trend extraction using moving average filtering
- **SeriesDecomposition** - Trend-seasonal decomposition using moving average
- **DFTSeriesDecomposition** - Frequency-based decomposition using Discrete Fourier Transform
- **ReversibleInstanceNorm** - Reversible instance normalization with optional denormalization
- **ReversibleInstanceNormMultivariate** - Multivariate reversible instance normalization
- **MultiScaleSeasonMixing** - Bottom-up multi-scale seasonal pattern mixing
- **MultiScaleTrendMixing** - Top-down multi-scale trend pattern mixing
- **PastDecomposableMixing** - Decomposable mixing encoder combining decomposition and multi-scale mixing
- **TemporalMixing** - MLP-based temporal mixing for TSMixer architecture
- **FeatureMixing** - Feed-forward feature mixing for cross-series correlations
- **MixingLayer** - Core mixing block combining temporal and feature mixing

### 🧠 Attention Mechanisms (6 layers)

**Advanced attention layers for capturing complex feature relationships and dependencies in tabular data.**

- **TabularAttention** - Dual attention mechanism for inter-feature and inter-sample relationships
- **MultiResolutionTabularAttention** - Multi-resolution attention for different feature scales
- **InterpretableMultiHeadAttention** - Multi-head attention with explainability features
- **TransformerBlock** - Standard transformer block with self-attention and feed-forward
- **ColumnAttention** - Column-wise attention for feature relationships
- **RowAttention** - Row-wise attention for sample relationships

### 🔧 Data Preprocessing & Transformation (9 layers)

**Essential preprocessing layers for data cleaning, transformation, and preparation for optimal model performance.**

- **DifferentiableTabularPreprocessor** - End-to-end differentiable preprocessing with learnable imputation
- **DifferentialPreprocessingLayer** - Multiple candidate transformations with learnable combination
- **DateParsingLayer** - Flexible date parsing from various formats
- **DateEncodingLayer** - Cyclical date feature encoding
- **SeasonLayer** - Seasonal feature extraction
- **DistributionTransformLayer** - Automatic distribution transformation
- **DistributionAwareEncoder** - Distribution-aware feature encoding
- **CastToFloat32Layer** - Type casting utility
- **AdvancedNumericalEmbedding** - Advanced numerical embedding with dual-branch architecture

### ⚙️ Feature Engineering & Selection (5 layers)

**Intelligent feature engineering and selection layers for identifying important features and creating powerful representations.**

- **VariableSelection** - Intelligent variable selection using gated residual networks
- **GatedFeatureSelection** - Learnable feature selection with gating
- **GatedFeatureFusion** - Gated mechanism for feature fusion
- **SparseAttentionWeighting** - Sparse attention for efficient computation
- **FeatureCutout** - Feature cutout for data augmentation and regularization

### 🏗️ Specialized Architectures (8 layers)

**Advanced specialized layers for specific use cases including gated networks, boosting, business rules, and ensemble methods.**

- **GatedResidualNetwork** - Gated residual network with improved gradient flow
- **GatedLinearUnit** - Gated linear transformation
- **TabularMoELayer** - Mixture of Experts for adaptive expert selection
- **BoostingBlock** - Gradient boosting inspired neural block
- **BoostingEnsembleLayer** - Ensemble of boosting blocks
- **BusinessRulesLayer** - Domain-specific business rules integration
- **StochasticDepth** - Stochastic depth regularization
- **SlowNetwork** - Careful feature processing with controlled information flow

### 🛠️ Utility & Graph Processing (8 layers)

**Essential utility layers for data processing, graph operations, and anomaly detection.**

- **GraphFeatureAggregation** - Graph feature aggregation for relational learning
- **AdvancedGraphFeatureLayer** - Advanced graph feature processing
- **MultiHeadGraphFeaturePreprocessor** - Multi-head graph preprocessing
- **NumericalAnomalyDetection** - Statistical anomaly detection for numerical features
- **CategoricalAnomalyDetectionLayer** - Pattern-based anomaly detection for categorical features
- **HyperZZWOperator** - Hyperparameter-aware operator for adaptive behavior

---

## 📋 Complete API Reference

<div class="api-reference-section">
  <div class="section-header">
    <h2>⏱️ Time Series & Forecasting (16 layers)</h2>
    <p>Specialized layers for time series analysis, forecasting, and pattern recognition with advanced decomposition and mixing strategies.</p>
  </div>

  <div class="api-grid">
    <div class="api-card">
      <h4>📍 PositionalEmbedding</h4>
      <div class="api-signature">PositionalEmbedding(max_len, embedding_dim)</div>
      <p>Sinusoidal positional encoding for sequence models and transformers.</p>
      <p><strong>Use when:</strong> You need position information in transformer models</p>
    </div>

    <div class="api-card">
      <h4>🔧 FixedEmbedding</h4>
      <div class="api-signature">FixedEmbedding(num_embeddings, embedding_dim)</div>
      <p>Non-trainable sinusoidal embeddings for temporal indices (months, days, hours).</p>
      <p><strong>Use when:</strong> You want fixed cyclical embeddings for temporal features</p>
    </div>

    <div class="api-card">
      <h4>🎫 TokenEmbedding</h4>
      <div class="api-signature">TokenEmbedding(c_in, d_model, conv_kernel_size)</div>
      <p>1D convolution-based embedding for time series values.</p>
      <p><strong>Use when:</strong> You need learnable embeddings for raw time series values</p>
    </div>

    <div class="api-card">
      <h4>⏰ TemporalEmbedding</h4>
      <div class="api-signature">TemporalEmbedding(d_model, embed_type, freq)</div>
      <p>Embedding layer for temporal features like month, day, weekday, hour, minute.</p>
      <p><strong>Use when:</strong> You have temporal feature information to encode</p>
    </div>

    <div class="api-card">
      <h4>🎯 DataEmbeddingWithoutPosition</h4>
      <div class="api-signature">DataEmbeddingWithoutPosition(c_in, d_model, embedding_type, freq, dropout)</div>
      <p>Combined token and temporal embedding for comprehensive feature representation.</p>
      <p><strong>Use when:</strong> You want unified embeddings for both values and temporal features</p>
    </div>

    <div class="api-card">
      <h4>🏃 MovingAverage</h4>
      <div class="api-signature">MovingAverage(kernel_size)</div>
      <p>Trend extraction using moving average filtering for time series.</p>
      <p><strong>Use when:</strong> You need to separate trends from seasonal components</p>
    </div>

    <div class="api-card">
      <h4>🔀 SeriesDecomposition</h4>
      <div class="api-signature">SeriesDecomposition(kernel_size)</div>
      <p>Trend-seasonal decomposition using moving average filtering.</p>
      <p><strong>Use when:</strong> You want explicit decomposition of time series components</p>
    </div>

    <div class="api-card">
      <h4>📊 DFTSeriesDecomposition</h4>
      <div class="api-signature">DFTSeriesDecomposition()</div>
      <p>Frequency-based series decomposition using Discrete Fourier Transform.</p>
      <p><strong>Use when:</strong> You prefer frequency-domain decomposition</p>
    </div>

    <div class="api-card">
      <h4>🔄 ReversibleInstanceNorm</h4>
      <div class="api-signature">ReversibleInstanceNorm(eps, subtract_last)</div>
      <p>Reversible instance normalization with optional denormalization for time series.</p>
      <p><strong>Use when:</strong> You need reversible normalization for stable training</p>
    </div>

    <div class="api-card">
      <h4>🏗️ ReversibleInstanceNormMultivariate</h4>
      <div class="api-signature">ReversibleInstanceNormMultivariate(eps)</div>
      <p>Multivariate version of reversible instance normalization.</p>
      <p><strong>Use when:</strong> You have multivariate time series data</p>
    </div>

    <div class="api-card">
      <h4>🌊 MultiScaleSeasonMixing</h4>
      <div class="api-signature">MultiScaleSeasonMixing(seq_len, down_sampling_window, d_model)</div>
      <p>Bottom-up multi-scale seasonal pattern mixing with hierarchical aggregation.</p>
      <p><strong>Use when:</strong> You want to capture seasonal patterns at multiple scales</p>
    </div>

    <div class="api-card">
      <h4>📈 MultiScaleTrendMixing</h4>
      <div class="api-signature">MultiScaleTrendMixing(seq_len, down_sampling_window, d_model)</div>
      <p>Top-down multi-scale trend pattern mixing with hierarchical decomposition.</p>
      <p><strong>Use when:</strong> You want to capture trend patterns at multiple scales</p>
    </div>

    <div class="api-card">
      <h4>🔀 PastDecomposableMixing</h4>
      <div class="api-signature">PastDecomposableMixing(seq_len, pred_len, d_model, decomp_method, down_sampling_window)</div>
      <p>Past decomposable mixing encoder combining decomposition and multi-scale mixing.</p>
      <p><strong>Use when:</strong> You need comprehensive decomposition with multi-scale mixing</p>
    </div>

    <div class="api-card">
      <h4>⏱️ TemporalMixing</h4>
      <div class="api-signature">TemporalMixing(seq_len, d_model, hidden_dim, dropout)</div>
      <p>MLP-based temporal mixing for TSMixer that applies transformations across time.</p>
      <p><strong>Use when:</strong> You want lightweight temporal pattern learning</p>
    </div>

    <div class="api-card">
      <h4>🔀 FeatureMixing</h4>
      <div class="api-signature">FeatureMixing(d_model, ff_dim, dropout)</div>
      <p>Feed-forward feature mixing learning cross-series correlations.</p>
      <p><strong>Use when:</strong> You want to capture dependencies between time series</p>
    </div>

    <div class="api-card">
      <h4>🔀 MixingLayer</h4>
      <div class="api-signature">MixingLayer(seq_len, d_model, hidden_dim, ff_dim, dropout)</div>
      <p>Core mixing block combining TemporalMixing and FeatureMixing for TSMixer.</p>
      <p><strong>Use when:</strong> You need dual-perspective temporal and feature learning</p>
    </div>
  </div>
</div>

<div class="api-reference-section">
  <div class="section-header">
    <h2>🎯 Feature Selection & Gating (5 layers)</h2>
    <p>Layers for dynamic feature selection, gating mechanisms, and feature fusion.</p>
  </div>

  <div class="api-grid">
    <div class="api-card">
      <h4>🔀 VariableSelection</h4>
      <div class="api-signature">VariableSelection(nr_features, units, dropout_rate)</div>
      <p>Dynamic feature selection using gated residual networks with optional context conditioning.</p>
      <p><strong>Use when:</strong> You need automatic feature importance learning during training</p>
    </div>

    <div class="api-card">
      <h4>🚪 GatedFeatureSelection</h4>
      <div class="api-signature">GatedFeatureSelection(units, dropout_rate)</div>
      <p>Feature selection layer using gating mechanisms for conditional feature routing.</p>
      <p><strong>Use when:</strong> You want learnable adaptive feature importance</p>
    </div>

    <div class="api-card">
      <h4>🌊 GatedFeatureFusion</h4>
      <div class="api-signature">GatedFeatureFusion(hidden_dim, dropout)</div>
      <p>Combines and fuses features using gated mechanisms for adaptive integration.</p>
      <p><strong>Use when:</strong> You need to intelligently combine multiple feature representations</p>
    </div>

    <div class="api-card">
      <h4>📍 GatedLinearUnit</h4>
      <div class="api-signature">GatedLinearUnit(units)</div>
      <p>Gated linear transformation for controlling information flow.</p>
      <p><strong>Use when:</strong> You need selective information flow in your model</p>
    </div>

    <div class="api-card">
      <h4>🔗 GatedResidualNetwork</h4>
      <div class="api-signature">GatedResidualNetwork(units, dropout_rate)</div>
      <p>Gated residual network architecture with improved gradient flow.</p>
      <p><strong>Use when:</strong> You need robust feature processing with residual connections</p>
    </div>
  </div>
</div>

<div class="api-reference-section">
  <div class="section-header">
    <h2>👁️ Attention Mechanisms (6 layers)</h2>
    <p>Advanced attention layers for capturing complex feature and sample relationships.</p>
  </div>

  <div class="api-grid">
    <div class="api-card">
      <h4>🎯 TabularAttention</h4>
      <div class="api-signature">TabularAttention(num_heads, key_dim, dropout)</div>
      <p>Dual attention mechanism for inter-feature and inter-sample relationships.</p>
      <p><strong>Use when:</strong> You have complex feature interactions to discover</p>
    </div>

    <div class="api-card">
      <h4>📊 MultiResolutionTabularAttention</h4>
      <div class="api-signature">MultiResolutionTabularAttention(num_heads, key_dim, dropout)</div>
      <p>Multi-resolution attention for numerical and categorical features.</p>
      <p><strong>Use when:</strong> You have mixed feature types needing different processing</p>
    </div>

    <div class="api-card">
      <h4>🔍 InterpretableMultiHeadAttention</h4>
      <div class="api-signature">InterpretableMultiHeadAttention(num_heads, key_dim, dropout)</div>
      <p>Multi-head attention with explainability features.</p>
      <p><strong>Use when:</strong> You need to understand attention patterns</p>
    </div>

    <div class="api-card">
      <h4>🧠 TransformerBlock</h4>
      <div class="api-signature">TransformerBlock(dim_model, num_heads, ff_units, dropout)</div>
      <p>Complete transformer block with self-attention and feed-forward.</p>
      <p><strong>Use when:</strong> You want standard transformer architecture for tabular data</p>
    </div>

    <div class="api-card">
      <h4>📌 ColumnAttention</h4>
      <div class="api-signature">ColumnAttention(hidden_dim, dropout)</div>
      <p>Column-wise attention for feature relationships.</p>
      <p><strong>Use when:</strong> You want to focus on feature-level interactions</p>
    </div>

    <div class="api-card">
      <h4>📍 RowAttention</h4>
      <div class="api-signature">RowAttention(hidden_dim, dropout)</div>
      <p>Row-wise attention for sample relationships.</p>
      <p><strong>Use when:</strong> You want to capture sample-level patterns</p>
    </div>
  </div>
</div>

<div class="api-reference-section">
  <div class="section-header">
    <h2>📊 Data Preprocessing & Transformation (9 layers)</h2>
    <p>Essential preprocessing layers for data preparation and transformation.</p>
  </div>

  <div class="api-grid">
    <div class="api-card">
      <h4>🔄 DistributionTransformLayer</h4>
      <div class="api-signature">DistributionTransformLayer(transform_type, epsilon, method)</div>
      <p>Automatic distribution transformation for improved analysis.</p>
      <p><strong>Use when:</strong> You have skewed distributions that need normalization</p>
    </div>

    <div class="api-card">
      <h4>🎓 DistributionAwareEncoder</h4>
      <div class="api-signature">DistributionAwareEncoder(encoding_dim, dropout, detection_method)</div>
      <p>Distribution-aware feature encoding with auto-detection.</p>
      <p><strong>Use when:</strong> You need adaptive encoding based on data distributions</p>
    </div>

    <div class="api-card">
      <h4>📈 AdvancedNumericalEmbedding</h4>
      <div class="api-signature">AdvancedNumericalEmbedding(embedding_dim, num_bins, hidden_dim)</div>
      <p>Advanced numerical embedding with dual-branch architecture.</p>
      <p><strong>Use when:</strong> You want rich numerical feature representations</p>
    </div>

    <div class="api-card">
      <h4>📅 DateParsingLayer</h4>
      <div class="api-signature">DateParsingLayer(date_formats, default_format)</div>
      <p>Flexible date parsing from various formats.</p>
      <p><strong>Use when:</strong> You have date/time features to parse</p>
    </div>

    <div class="api-card">
      <h4>🕐 DateEncodingLayer</h4>
      <div class="api-signature">DateEncodingLayer(min_year, max_year)</div>
      <p>Cyclical date feature encoding.</p>
      <p><strong>Use when:</strong> You want cyclical representations of temporal features</p>
    </div>

    <div class="api-card">
      <h4>🌙 SeasonLayer</h4>
      <div class="api-signature">SeasonLayer()</div>
      <p>Seasonal feature extraction for temporal patterns.</p>
      <p><strong>Use when:</strong> Your data has seasonal patterns</p>
    </div>

    <div class="api-card">
      <h4>🔀 DifferentialPreprocessingLayer</h4>
      <div class="api-signature">DifferentialPreprocessingLayer(transform_types, dropout)</div>
      <p>Multiple transformations with learnable combination.</p>
      <p><strong>Use when:</strong> You want the model to learn optimal preprocessing</p>
    </div>

    <div class="api-card">
      <h4>🔧 DifferentiableTabularPreprocessor</h4>
      <div class="api-signature">DifferentiableTabularPreprocessor(imputation_strategy, normalization, dropout)</div>
      <p>End-to-end differentiable preprocessing.</p>
      <p><strong>Use when:</strong> You want learnable imputation and normalization</p>
    </div>

    <div class="api-card">
      <h4>🎨 CastToFloat32Layer</h4>
      <div class="api-signature">CastToFloat32Layer()</div>
      <p>Type casting utility for float32 precision.</p>
      <p><strong>Use when:</strong> You need to ensure consistent data types</p>
    </div>
  </div>
</div>

<div class="api-reference-section">
  <div class="section-header">
    <h2>⚙️ Feature Engineering & Selection (5 layers)</h2>
    <p>Advanced feature engineering and selection layers.</p>
  </div>

  <div class="api-grid">
    <div class="api-card">
      <h4>🧬 GraphFeatureAggregation</h4>
      <div class="api-signature">GraphFeatureAggregation(aggregation_method, hidden_dim, dropout)</div>
      <p>Graph feature aggregation for relational learning.</p>
      <p><strong>Use when:</strong> You have feature relationships to model</p>
    </div>

    <div class="api-card">
      <h4>🎯 SparseAttentionWeighting</h4>
      <div class="api-signature">SparseAttentionWeighting(temperature, dropout, sparsity_threshold)</div>
      <p>Sparse attention for efficient computation.</p>
      <p><strong>Use when:</strong> You need memory-efficient attention</p>
    </div>

    <div class="api-card">
      <h4>🗑️ FeatureCutout</h4>
      <div class="api-signature">FeatureCutout(cutout_prob, noise_value, training_only)</div>
      <p>Feature cutout for data augmentation.</p>
      <p><strong>Use when:</strong> You want to improve model robustness through augmentation</p>
    </div>
  </div>
</div>

<div class="api-reference-section">
  <div class="section-header">
    <h2>🏗️ Specialized Architectures (8 layers)</h2>
    <p>Advanced specialized layers for specific use cases.</p>
  </div>

  <div class="api-grid">
    <div class="api-card">
      <h4>📈 BoostingBlock</h4>
      <div class="api-signature">BoostingBlock(hidden_units, hidden_activation, gamma_trainable)</div>
      <p>Gradient boosting inspired neural block.</p>
      <p><strong>Use when:</strong> You want boosting-like behavior in neural networks</p>
    </div>

    <div class="api-card">
      <h4>🎯 BoostingEnsembleLayer</h4>
      <div class="api-signature">BoostingEnsembleLayer(num_learners, learner_units, hidden_activation)</div>
      <p>Ensemble of boosting blocks.</p>
      <p><strong>Use when:</strong> You want ensemble-based learning</p>
    </div>

    <div class="api-card">
      <h4>🏗️ BusinessRulesLayer</h4>
      <div class="api-signature">BusinessRulesLayer(rules, feature_type, trainable_weights)</div>
      <p>Domain-specific business rules integration.</p>
      <p><strong>Use when:</strong> You need to enforce domain constraints</p>
    </div>

    <div class="api-card">
      <h4>🐢 SlowNetwork</h4>
      <div class="api-signature">SlowNetwork(hidden_units, num_layers, activation, dropout)</div>
      <p>Careful feature processing with controlled flow.</p>
      <p><strong>Use when:</strong> You want deliberate, well-controlled processing</p>
    </div>

    <div class="api-card">
      <h4>⚡ HyperZZWOperator</h4>
      <div class="api-signature">HyperZZWOperator(hidden_units, hyperparameter_dim, activation)</div>
      <p>Hyperparameter-aware operator for adaptive behavior.</p>
      <p><strong>Use when:</strong> You want dynamic hyperparameter adjustment</p>
    </div>

    <div class="api-card">
      <h4>📊 TabularMoELayer</h4>
      <div class="api-signature">TabularMoELayer(num_experts, expert_units)</div>
      <p>Mixture of Experts for tabular data.</p>
      <p><strong>Use when:</strong> You have diverse data requiring different expert processing</p>
    </div>

    <div class="api-card">
      <h4>🎲 StochasticDepth</h4>
      <div class="api-signature">StochasticDepth(survival_prob, scale_at_test)</div>
      <p>Stochastic depth regularization.</p>
      <p><strong>Use when:</strong> You want improved generalization in deep networks</p>
    </div>
  </div>
</div>

<div class="api-reference-section">
  <div class="section-header">
    <h2>🛠️ Utility & Graph Processing (8 layers)</h2>
    <p>Utility layers for data processing, graph operations, and anomaly detection.</p>
  </div>

  <div class="api-grid">
    <div class="api-card">
      <h4>🧬 AdvancedGraphFeatureLayer</h4>
      <div class="api-signature">AdvancedGraphFeatureLayer(hidden_dim, num_heads, dropout, use_attention)</div>
      <p>Advanced graph feature processing with dynamic learning.</p>
      <p><strong>Use when:</strong> You have complex feature relationships</p>
    </div>

    <div class="api-card">
      <h4>👥 MultiHeadGraphFeaturePreprocessor</h4>
      <div class="api-signature">MultiHeadGraphFeaturePreprocessor(num_heads, hidden_dim, dropout, aggregation)</div>
      <p>Multi-head graph preprocessing.</p>
      <p><strong>Use when:</strong> You want parallel feature processing</p>
    </div>

    <div class="api-card">
      <h4>📉 NumericalAnomalyDetection</h4>
      <div class="api-signature">NumericalAnomalyDetection(method, contamination, threshold)</div>
      <p>Statistical anomaly detection for numerical features.</p>
      <p><strong>Use when:</strong> You need to detect numerical outliers</p>
    </div>

    <div class="api-card">
      <h4>📊 CategoricalAnomalyDetectionLayer</h4>
      <div class="api-signature">CategoricalAnomalyDetectionLayer(method, threshold, min_frequency)</div>
      <p>Pattern-based anomaly detection for categorical features.</p>
      <p><strong>Use when:</strong> You need to detect categorical anomalies</p>
    </div>
  </div>
</div>

---

## 🚀 Quick Start Guide

<div class="highlight-box">
  <h3>Getting Started with KerasFactory Layers</h3>
  
  **Step 1: Choose Your Base Layer**
  - Start with `DifferentiableTabularPreprocessor` for data preparation
  - Add `VariableSelection` for feature importance
  
  **Step 2: Add Attention**
  - Use `TabularAttention` to capture feature relationships
  
  **Step 3: Build Your Model**
  - Stack layers together for powerful architectures
  
  **Example:**
  ```python
  import keras
  from kerasfactory.layers import TabularAttention, VariableSelection
  
  inputs = keras.Input(shape=(10,))
  x = VariableSelection(nr_features=10, units=64)(inputs)
  x = TabularAttention(num_heads=4, key_dim=32)(x)
  outputs = keras.layers.Dense(1, activation='sigmoid')(x)
  
  model = keras.Model(inputs, outputs)
  ```
</div>

---

## 📖 For More Information

- **[API Reference](api/layers.md)** - Detailed API documentation with autodoc references
- **[Contributing](contributing.md)** - How to contribute new layers
- **[Examples](../examples/README.md)** - Real-world usage examples
- **[Tutorials](../tutorials/basic-workflows.md)** - Step-by-step guides
