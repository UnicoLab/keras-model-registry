---
title: Layers API Reference - KMR
description: Complete reference for 38+ production-ready KMR layers including attention mechanisms, feature engineering, preprocessing, and specialized architectures for tabular data.
keywords: keras layers, tabular data layers, attention mechanisms, feature engineering, preprocessing layers, neural network layers
---

<style>
.api-hero-kdp {
  background: linear-gradient(135deg, var(--md-primary-fg-color) 0%, var(--md-primary-fg-color-light) 100%);
  color: white;
  padding: 4rem 2rem;
  border-radius: 16px;
  margin-bottom: 3rem;
  text-align: center;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.api-hero-kdp h1 {
  font-size: 3rem;
  margin-bottom: 1rem;
  font-weight: 700;
  letter-spacing: -1px;
}

.api-tagline {
  font-size: 1.3rem;
  margin-bottom: 2rem;
  opacity: 0.95;
  line-height: 1.6;
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
}

.api-stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.2);
}

.api-stat-item {
  text-align: center;
}

.api-stat-value {
  font-size: 2.5rem;
  font-weight: 700;
  display: block;
  margin-bottom: 0.5rem;
}

.api-stat-label {
  font-size: 0.95rem;
  opacity: 0.9;
  font-weight: 500;
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

.highlight-box strong {
  color: var(--md-primary-fg-color);
}

.quick-nav-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.nav-card {
  background: var(--md-default-bg-color);
  padding: 1.5rem;
  border-radius: 12px;
  border-left: 4px solid var(--md-primary-fg-color);
  box-shadow: var(--md-shadow-z1);
  transition: all 0.3s ease;
}

.nav-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--md-shadow-z2);
}

.nav-card h3 {
  margin: 0 0 1rem 0;
  color: var(--md-primary-fg-color);
}

.nav-card ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.nav-card li {
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--md-default-fg-color-10);
}

.nav-card li:last-child {
  border-bottom: none;
}

.nav-card a {
  color: var(--md-default-fg-color);
  text-decoration: none;
  transition: color 0.2s ease;
}

.nav-card a:hover {
  color: var(--md-primary-fg-color);
  font-weight: 500;
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

@media (max-width: 768px) {
  .api-hero-kdp h1 {
    font-size: 2rem;
  }
  
  .api-tagline {
    font-size: 1.1rem;
  }
  
  .api-stats-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
  }
  
  .quick-nav-grid {
    grid-template-columns: 1fr;
  }
  
  .feature-highlights {
    grid-template-columns: 1fr;
  }
}
</style>

<div class="api-hero-kdp">
  <h1>🧩 Layers API Reference</h1>
  <div class="api-tagline">
    <strong>38+ production-ready layers</strong> designed exclusively for <strong>Keras 3</strong>.<br>
    Build sophisticated tabular models with advanced attention, feature engineering, and preprocessing layers.
  </div>
  
  <div class="api-stats-grid">
    <div class="api-stat-item">
      <span class="api-stat-value">38+</span>
      <span class="api-stat-label">Production Layers</span>
    </div>
    <div class="api-stat-item">
      <span class="api-stat-value">5</span>
      <span class="api-stat-label">Categories</span>
    </div>
    <div class="api-stat-item">
      <span class="api-stat-value">100%</span>
      <span class="api-stat-label">Keras 3 Native</span>
    </div>
    <div class="api-stat-item">
      <span class="api-stat-value">0%</span>
      <span class="api-stat-label">TensorFlow Lock-in</span>
    </div>
  </div>
</div>

## 🎯 Why Use KMR Layers?

| Challenge | Traditional Approach | KMR's Solution |
|-----------|----------------------|----------------|
| 🔗 **Feature Interactions** | Manual feature crosses | 👁️ **Tabular Attention** - Automatic relationship discovery |
| 🏷️ **Mixed Feature Types** | Uniform processing | 🧩 **Feature-wise Layers** - Specialized processing per feature |
| 📊 **Complex Distributions** | Fixed strategies | 📊 **Distribution-Aware Encoding** - Adaptive transformations |
| ⚡ **Performance Optimization** | Post-hoc analysis | 🎯 **Built-in Selection** - Learned during training |
| 🔒 **Production Readiness** | Extra tooling needed | ✅ **Battle-Tested** - Used in production models |

## 🚀 Quick Navigation

<div class="quick-nav-grid">
  <div class="nav-card">
    <h3>🧠 Attention Layers</h3>
    <ul>
      <li><a href="#attention-layers">5 layers for feature relationships</a></li>
      <li><a href="../../layers/tabular-attention/">TabularAttention</a></li>
      <li><a href="../../layers/multi-resolution-tabular-attention/">MultiResolution</a></li>
      <li><a href="../../layers/column-attention/">Column & Row Attention</a></li>
      <li><a href="../../layers/interpretable-multi-head-attention/">Interpretable Heads</a></li>
    </ul>
  </div>
  
  <div class="nav-card">
    <h3>🔧 Preprocessing Layers</h3>
    <ul>
      <li><a href="#preprocessing-layers">6 layers for data prep</a></li>
      <li><a href="../../layers/differentiable-tabular-preprocessor/">Tabular Preprocessor</a></li>
      <li><a href="../../layers/date-encoding-layer/">Date Encoding</a></li>
      <li><a href="../../layers/season-layer/">Season Layer</a></li>
      <li><a href="../../layers/cast-to-float32-layer/">Type Casting</a></li>
    </ul>
  </div>
  
  <div class="nav-card">
    <h3>⚙️ Feature Engineering</h3>
    <ul>
      <li><a href="#feature-engineering-layers">8 layers for features</a></li>
      <li><a href="../../layers/variable-selection/">Variable Selection</a></li>
      <li><a href="../../layers/advanced-numerical-embedding/">Advanced Embedding</a></li>
      <li><a href="../../layers/distribution-aware-encoder/">Distribution Aware</a></li>
      <li><a href="../../layers/feature-cutout/">Feature Cutout</a></li>
    </ul>
  </div>
  
  <div class="nav-card">
    <h3>🏗️ Specialized Architectures</h3>
    <ul>
      <li><a href="#specialized-layers">11 advanced layers</a></li>
      <li><a href="../../layers/gated-residual-network/">Gated Residual Network</a></li>
      <li><a href="../../layers/transformer-block/">Transformer Block</a></li>
      <li><a href="../../layers/tabular-moe-layer/">Mixture of Experts</a></li>
      <li><a href="../../layers/boosting-ensemble-layer/">Boosting Ensemble</a></li>
    </ul>
  </div>
  
  <div class="nav-card">
    <h3>🛠️ Utility Layers</h3>
    <ul>
      <li><a href="#utility-layers">8 utility layers</a></li>
      <li><a href="../../layers/advanced-graph-feature/">Graph Features</a></li>
      <li><a href="../../layers/numerical-anomaly-detection/">Anomaly Detection</a></li>
      <li><a href="../../layers/text-preprocessing-layer/">Text Processing</a></li>
      <li><a href="../../layers/multi-head-graph-feature-preprocessor/">Graph Preprocessing</a></li>
    </ul>
  </div>
</div>

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

## 🧠 Attention Layers {#attention-layers}

<div class="layers-section">
  <div class="section-header">
    <h2>🧠 Attention Mechanisms</h2>
    <p>Advanced attention layers for capturing complex feature relationships and dependencies in tabular data.</p>
    <div class="section-stats">
      <span class="stat">5 layers</span>
      <span class="stat">High Performance</span>
      <span class="stat">Complex Interactions</span>
    </div>
  </div>
  
  <div class="layer-reference-grid">
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>TabularAttention</h3>
        <div class="layer-badges">
          <span class="badge badge-popular">🔥 Popular</span>
          <span class="badge badge-stable">✅ Stable</span>
      </div>
    </div>
      <p class="layer-description">Dual attention mechanism for inter-feature and inter-sample relationships in tabular data.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 💾 Memory Efficient | 🎯 High Accuracy</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/tabular-attention.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#tabularattention-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>MultiResolutionTabularAttention</h3>
        <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
          <span class="badge badge-stable">✅ Stable</span>
      </div>
    </div>
      <p class="layer-description">Multi-resolution attention mechanism that processes numerical and categorical features separately.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 High Accuracy | 💾 Memory Efficient</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/multi-resolution-tabular-attention.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#multiresolutiontabularattention-api" class="action-btn secondary">📋 API</a>
      </div>
  </div>
  
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>ColumnAttention</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
    </div>
  </div>
      <p class="layer-description">Column-wise attention for tabular data to capture feature-level relationships.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Interpretable</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/column-attention.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#columnattention-api" class="action-btn secondary">📋 API</a>
      </div>
</div>

    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
    <div class="layer-header">
        <h3>RowAttention</h3>
      <div class="layer-badges">
        <span class="badge badge-stable">✅ Stable</span>
      </div>
    </div>
      <p class="layer-description">Row-wise attention mechanisms for sample-level pattern recognition.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Sample Relationships</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/row-attention.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#rowattention-api" class="action-btn secondary">📋 API</a>
    </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>InterpretableMultiHeadAttention</h3>
        <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Interpretable multi-head attention with attention weight analysis and visualization.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 Interpretable | 🔍 Explainable</div>
      </div>
    <div class="layer-actions">
        <a href="../layers/interpretable-multi-head-attention.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#interpretablemultiheadattention-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    </div>
  </div>
  
## 🔧 Preprocessing Layers {#preprocessing-layers}

<div class="layers-section">
  <div class="section-header">
    <h2>🔧 Preprocessing & Data Transformation</h2>
    <p>Essential preprocessing layers for data cleaning, transformation, and preparation for optimal model performance.</p>
    <div class="section-stats">
      <span class="stat">6 layers</span>
      <span class="stat">Beginner Friendly</span>
      <span class="stat">Data Quality</span>
    </div>
  </div>

  <div class="layer-reference-grid">
    <div class="layer-reference-card" data-complexity="beginner" data-use-case="tabular">
    <div class="layer-header">
        <h3>DifferentiableTabularPreprocessor</h3>
      <div class="layer-badges">
        <span class="badge badge-popular">🔥 Popular</span>
        <span class="badge badge-stable">✅ Stable</span>
      </div>
    </div>
      <p class="layer-description">End-to-end differentiable preprocessing for tabular data with learnable imputation and normalization.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Learnable | 🔧 End-to-End</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/differentiable-tabular-preprocessor.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#differentiabletabularpreprocessor-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>DifferentialPreprocessingLayer</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Advanced preprocessing with multiple candidate transformations and learnable combination.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 Adaptive | 🔧 Multiple Transforms</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/differential-preprocessing-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#differentialpreprocessinglayer-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="beginner" data-use-case="tabular">
      <div class="layer-header">
        <h3>DateParsingLayer</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Flexible date parsing and extraction from various date formats and strings.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, 1) - date strings</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, 4) - [year, month, day, dow]</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 📅 Date Processing</div>
      </div>
    <div class="layer-actions">
        <a href="../layers/date-parsing-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#dateparsinglayer-api" class="action-btn secondary">📋 API</a>
    </div>
  </div>
  
    <div class="layer-reference-card" data-complexity="beginner" data-use-case="tabular">
    <div class="layer-header">
        <h3>DateEncodingLayer</h3>
      <div class="layer-badges">
        <span class="badge badge-popular">🔥 Popular</span>
        <span class="badge badge-stable">✅ Stable</span>
      </div>
    </div>
      <p class="layer-description">Comprehensive date and time feature encoding with cyclical representations.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, 4) - [year, month, day, dow]</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, 8) - cyclical encodings</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 📅 Cyclical Encoding</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/date-encoding-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#dateencodinglayer-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="beginner" data-use-case="tabular">
      <div class="layer-header">
        <h3>SeasonLayer</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Seasonal feature extraction from date/time data for temporal pattern recognition.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, 4) - [year, month, day, dow]</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, 8) - original + 4 seasons</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🌸 Seasonal Patterns</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/season-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#seasonlayer-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="beginner" data-use-case="tabular">
      <div class="layer-header">
        <h3>CastToFloat32Layer</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Type casting utility layer for ensuring consistent data types throughout the model.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> Any numeric tensor</div>
        <div class="info-item"><strong>Output:</strong> float32 tensor</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🔧 Utility</div>
      </div>
    <div class="layer-actions">
        <a href="../layers/cast-to-float32-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#casttofloat32layer-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
  </div>
</div>

## ⚙️ Feature Engineering Layers {#feature-engineering-layers}

<div class="layers-section">
  <div class="section-header">
    <h2>⚙️ Feature Engineering & Selection</h2>
    <p>Advanced feature engineering layers for intelligent feature selection, transformation, and representation learning.</p>
    <div class="section-stats">
      <span class="stat">8 layers</span>
      <span class="stat">Feature Intelligence</span>
      <span class="stat">High Performance</span>
    </div>
  </div>
  
  <div class="layer-reference-grid">
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>VariableSelection</h3>
        <div class="layer-badges">
          <span class="badge badge-popular">🔥 Popular</span>
          <span class="badge badge-stable">✅ Stable</span>
    </div>
      </div>
      <p class="layer-description">Intelligent variable selection network for identifying important features using gated residual networks.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Interpretable | 🔧 Feature Selection</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/variable-selection.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#variableselection-api" class="action-btn secondary">📋 API</a>
      </div>
  </div>
  
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>GatedFeatureSelection</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
    </div>
      </div>
      <p class="layer-description">Learnable feature selection with gating network and residual connection for adaptive feature importance.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Learnable | 🔧 Adaptive</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/gated-feature-selection.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#gatedfeatureselection-api" class="action-btn secondary">📋 API</a>
      </div>
  </div>
  
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>GatedFeatureFusion</h3>
        <div class="layer-badges">
          <span class="badge badge-popular">🔥 Popular</span>
          <span class="badge badge-stable">✅ Stable</span>
    </div>
      </div>
      <p class="layer-description">Gated mechanism for intelligently fusing multiple feature representations with learnable weights.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> List of feature tensors</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 💾 Memory Efficient | 🔧 Flexible</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/gated-feature-fusion.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#gatedfeaturefusion-api" class="action-btn secondary">📋 API</a>
      </div>
  </div>
  
    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>AdvancedNumericalEmbedding</h3>
        <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
          <span class="badge badge-stable">✅ Stable</span>
    </div>
      </div>
      <p class="layer-description">Advanced numerical feature embedding with dual-branch architecture for continuous and discrete features.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, embedding_dim)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 High Accuracy | 🔧 Flexible</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/advanced-numerical-embedding.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#advancednumericalembedding-api" class="action-btn secondary">📋 API</a>
  </div>
</div>

    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>DistributionAwareEncoder</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Distribution-aware feature encoding that automatically detects distribution type and applies appropriate transformations.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 Adaptive | 🔧 Auto Detection</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/distribution-aware-encoder.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#distributionawareencoder-api" class="action-btn secondary">📋 API</a>
      </div>
  </div>

    <div class="layer-reference-card" data-complexity="beginner" data-use-case="tabular">
      <div class="layer-header">
        <h3>DistributionTransformLayer</h3>
      <div class="layer-badges">
        <span class="badge badge-popular">🔥 Popular</span>
        <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Automatic distribution transformation for numerical features to improve model performance.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Accurate | 😊 Easy to Use</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/distribution-transform-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#distributiontransformlayer-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>SparseAttentionWeighting</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Sparse attention weighting mechanisms for efficient computation and selective feature combination.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> List of feature tensors</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 💾 Memory Efficient | 🔧 Sparse</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/sparse-attention-weighting.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#sparseattentionweighting-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>

    <div class="layer-reference-card" data-complexity="beginner" data-use-case="tabular">
      <div class="layer-header">
        <h3>FeatureCutout</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Feature cutout for data augmentation and regularization in tabular data by randomly masking features.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Regularization | 🔧 Augmentation</div>
        </div>
      <div class="layer-actions">
        <a href="../layers/feature-cutout.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#featurecutout-api" class="action-btn secondary">📋 API</a>
        </div>
        </div>
      </div>
        </div>
        
## 🏗️ Specialized Layers {#specialized-layers}

<div class="layers-section">
  <div class="section-header">
    <h2>🏗️ Specialized Architectures</h2>
    <p>Advanced specialized layers for specific use cases including gated networks, boosting, business rules, and anomaly detection.</p>
    <div class="section-stats">
      <span class="stat">11 layers</span>
      <span class="stat">Advanced</span>
      <span class="stat">Specialized Use Cases</span>
        </div>
        </div>
        
  <div class="layer-reference-grid">
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>GatedResidualNetwork</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Gated residual network combining residual connections with gated linear units for improved gradient flow.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, input_dim)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, units)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 High Accuracy | 🔧 Gradient Flow</div>
    </div>
      <div class="layer-actions">
        <a href="../layers/gated-residual-network.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#gatedresidualnetwork-api" class="action-btn secondary">📋 API</a>
    </div>
  </div>

    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>GatedLinearUnit</h3>
      <div class="layer-badges">
        <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Gated linear unit for intelligent feature gating and selective information flow control.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, input_dim)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, units)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Selective | 🔧 Gating</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/gated-linear-unit.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#gatedlinearunit-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>TransformerBlock</h3>
        <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Standard transformer block with multi-head attention and feed-forward networks for tabular data.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, dim_model) or (batch_size, seq_len, dim_model)</div>
        <div class="info-item"><strong>Output:</strong> Same as input</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 High Accuracy | 🔧 Flexible</div>
        </div>
      <div class="layer-actions">
        <a href="../layers/transformer-block.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#transformerblock-api" class="action-btn secondary">📋 API</a>
        </div>
        </div>

    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>TabularMoELayer</h3>
        <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Mixture of Experts for tabular data with adaptive expert selection and routing.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 High Accuracy | 🔧 Adaptive | 💾 Scalable</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/tabular-moe-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#tabularmoelayer-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>BoostingBlock</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Gradient boosting inspired neural network block for sequential learning and residual correction.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, input_dim)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, input_dim)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Boosting | 🔧 Sequential</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/boosting-block.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#boostingblock-api" class="action-btn secondary">📋 API</a>
    </div>
  </div>

    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>BoostingEnsembleLayer</h3>
      <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
        <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Ensemble of boosting blocks for improved performance and robustness through parallel weak learners.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, input_dim)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, input_dim)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 High Accuracy | 🔧 Ensemble | 💾 Parallel</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/boosting-ensemble-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#boostingensemblelayer-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>BusinessRulesLayer</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Integration of business rules and domain knowledge into neural networks for anomaly detection.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, 1)</div>
        <div class="info-item"><strong>Output:</strong> Dictionary with anomaly flags and violations</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Domain Knowledge | 🔧 Rules</div>
        </div>
      <div class="layer-actions">
        <a href="../layers/business-rules-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#businessruleslayer-api" class="action-btn secondary">📋 API</a>
        </div>
        </div>

    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>StochasticDepth</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Stochastic depth regularization for improved training and generalization in deep networks.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> Any tensor</div>
        <div class="info-item"><strong>Output:</strong> Same as input</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Better Generalization | 🔧 Regularization</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/stochastic-depth.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#stochasticdepth-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>SlowNetwork</h3>
        <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Slow network architecture for careful and deliberate feature processing with controlled information flow.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, input_dim)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, output_dim)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 High Accuracy | 🔧 Controlled | ⏱️ Deliberate</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/slow-network.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#slownetwork-api" class="action-btn secondary">📋 API</a>
    </div>
  </div>

    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>HyperZZWOperator</h3>
      <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
        <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Hyperparameter-aware operator for adaptive model behavior and dynamic parameter adjustment.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, input_dim)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, output_dim)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 Adaptive | 🔧 Hyperparameter-aware | ⚙️ Dynamic</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/hyper-zzw-operator.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#hyperzzwoperator-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>TextPreprocessingLayer</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Text preprocessing utilities for natural language features in tabular data with tokenization and encoding.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, 1) - text strings</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, max_length) - tokenized sequences</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 📝 Text Processing | 🔧 NLP</div>
        </div>
      <div class="layer-actions">
        <a href="../layers/text-preprocessing-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#textpreprocessinglayer-api" class="action-btn secondary">📋 API</a>
        </div>
        </div>
      </div>
    </div>
    
## 🛠️ Utility Layers {#utility-layers}

<div class="layers-section">
  <div class="section-header">
    <h2>🛠️ Utility & Graph Layers</h2>
    <p>Essential utility layers for data processing, graph operations, and anomaly detection in tabular data.</p>
    <div class="section-stats">
      <span class="stat">8 layers</span>
      <span class="stat">Essential</span>
      <span class="stat">Data Processing</span>
    </div>
  </div>

  <div class="layer-reference-grid">
    <div class="layer-reference-card" data-complexity="beginner" data-use-case="tabular">
      <div class="layer-header">
        <h3>CastToFloat32Layer</h3>
      <div class="layer-badges">
        <span class="badge badge-stable">✅ Stable</span>
      </div>
    </div>
      <p class="layer-description">Type casting utility layer for ensuring consistent data types throughout the model.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> Any numeric tensor</div>
        <div class="info-item"><strong>Output:</strong> float32 tensor</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🔧 Utility | 📊 Type Safety</div>
        </div>
      <div class="layer-actions">
        <a href="../layers/cast-to-float32-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#casttofloat32layer-api" class="action-btn secondary">📋 API</a>
        </div>
        </div>

    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>AdvancedGraphFeature</h3>
        <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
          <span class="badge badge-stable">✅ Stable</span>
      </div>
    </div>
      <p class="layer-description">Advanced graph-based feature processing with dynamic adjacency learning and relationship modeling.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 High Accuracy | 🔧 Graph Processing | 🕸️ Relationships</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/advanced-graph-feature.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#advancedgraphfeature-api" class="action-btn secondary">📋 API</a>
    </div>
  </div>

    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>GraphFeatureAggregation</h3>
      <div class="layer-badges">
        <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Graph feature aggregation mechanisms for relationship modeling and feature interaction learning.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Aggregation | 🕸️ Graph Features</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/graph-feature-aggregation.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#graphfeatureaggregation-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="advanced" data-use-case="tabular">
      <div class="layer-header">
        <h3>MultiHeadGraphFeaturePreprocessor</h3>
        <div class="layer-badges">
          <span class="badge badge-advanced">🚀 Advanced</span>
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Multi-head graph feature preprocessing for complex feature interactions and relationship learning.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Performance:</strong> 🎯 High Accuracy | 🔧 Multi-Head | 🕸️ Complex Interactions</div>
        </div>
      <div class="layer-actions">
        <a href="../layers/multi-head-graph-feature-preprocessor.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#multiheadgraphfeaturepreprocessor-api" class="action-btn secondary">📋 API</a>
        </div>
        </div>

    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>NumericalAnomalyDetection</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Anomaly detection for numerical features using statistical and machine learning methods.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> Dictionary with anomaly scores and flags</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Anomaly Detection | 🔍 Statistical</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/numerical-anomaly-detection.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#numericalanomalydetection-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
    
    <div class="layer-reference-card" data-complexity="intermediate" data-use-case="tabular">
      <div class="layer-header">
        <h3>CategoricalAnomalyDetectionLayer</h3>
        <div class="layer-badges">
          <span class="badge badge-stable">✅ Stable</span>
        </div>
      </div>
      <p class="layer-description">Anomaly detection for categorical features with pattern recognition and frequency analysis.</p>
      <div class="layer-quick-info">
        <div class="info-item"><strong>Input:</strong> (batch_size, num_features)</div>
        <div class="info-item"><strong>Output:</strong> Dictionary with anomaly scores and flags</div>
        <div class="info-item"><strong>Performance:</strong> ⚡ Fast | 🎯 Categorical | 🔍 Pattern Recognition</div>
      </div>
      <div class="layer-actions">
        <a href="../layers/categorical-anomaly-detection-layer.md" class="action-btn primary">📖 Full Docs</a>
        <a href="#categoricalanomalydetectionlayer-api" class="action-btn secondary">📋 API</a>
      </div>
    </div>
  </div>
</div>

## 📋 Complete API Reference

<div class="api-reference-section">
  <div class="section-header">
    <h2>📋 Complete API Reference</h2>
    <p>Detailed API documentation for all 38+ layers with parameters, methods, and examples.</p>
  </div>

  <div class="api-reference-tabs">
    <div class="tab-navigation">
      <button class="tab-btn active" data-tab="attention-api">🧠 Attention</button>
      <button class="tab-btn" data-tab="preprocessing-api">🔧 Preprocessing</button>
      <button class="tab-btn" data-tab="feature-engineering-api">⚙️ Feature Engineering</button>
      <button class="tab-btn" data-tab="specialized-api">🏗️ Specialized</button>
      <button class="tab-btn" data-tab="utility-api">🛠️ Utility</button>
    </div>

    <div class="tab-content active" id="attention-api">
      <div class="api-reference-grid">
        <div class="api-reference-card">
          <h3>TabularAttention</h3>
          <div class="api-signature">
            <code>TabularAttention(num_heads=8, key_dim=64, dropout=0.1, use_attention_weights=True, attention_activation='softmax')</code>
          </div>
          <div class="api-description">
            <p>Dual attention mechanism for inter-feature and inter-sample relationships in tabular data.</p>
          </div>
          <div class="api-parameters">
            <h4>Parameters</h4>
            <ul>
              <li><code>num_heads</code> (int): Number of attention heads</li>
              <li><code>key_dim</code> (int): Dimension of key vectors</li>
              <li><code>dropout</code> (float): Dropout rate</li>
              <li><code>use_attention_weights</code> (bool): Whether to return attention weights</li>
              <li><code>attention_activation</code> (str): Activation function for attention</li>
            </ul>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>MultiResolutionTabularAttention</h3>
          <div class="api-signature">
            <code>MultiResolutionTabularAttention(num_heads=8, key_dim=64, dropout=0.1, numerical_heads=4, categorical_heads=4)</code>
          </div>
          <div class="api-description">
            <p>Multi-resolution attention mechanism that processes numerical and categorical features separately.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>ColumnAttention</h3>
          <div class="api-signature">
            <code>ColumnAttention(hidden_dim=64, dropout=0.1, activation='relu')</code>
          </div>
          <div class="api-description">
            <p>Column-wise attention for tabular data to capture feature-level relationships.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>RowAttention</h3>
          <div class="api-signature">
            <code>RowAttention(hidden_dim=64, dropout=0.1, activation='relu')</code>
          </div>
          <div class="api-description">
            <p>Row-wise attention mechanisms for sample-level pattern recognition.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>InterpretableMultiHeadAttention</h3>
          <div class="api-signature">
            <code>InterpretableMultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1, return_attention_scores=True)</code>
          </div>
          <div class="api-description">
            <p>Interpretable multi-head attention with attention weight analysis and visualization.</p>
          </div>
        </div>
      </div>
    </div>

    <div class="tab-content" id="preprocessing-api">
      <div class="api-reference-grid">
        <div class="api-reference-card">
          <h3>DifferentiableTabularPreprocessor</h3>
          <div class="api-signature">
            <code>DifferentiableTabularPreprocessor(imputation_strategy='learnable', normalization='learnable', dropout=0.1)</code>
          </div>
          <div class="api-description">
            <p>End-to-end differentiable preprocessing for tabular data with learnable imputation and normalization.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>DifferentialPreprocessingLayer</h3>
          <div class="api-signature">
            <code>DifferentialPreprocessingLayer(transform_types=['identity', 'affine', 'mlp', 'log'], dropout=0.1)</code>
          </div>
          <div class="api-description">
            <p>Advanced preprocessing with multiple candidate transformations and learnable combination.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>DateParsingLayer</h3>
          <div class="api-signature">
            <code>DateParsingLayer(date_formats=None, default_format='%Y-%m-%d')</code>
          </div>
          <div class="api-description">
            <p>Flexible date parsing and extraction from various date formats and strings.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>DateEncodingLayer</h3>
          <div class="api-signature">
            <code>DateEncodingLayer(min_year=1900, max_year=2100)</code>
          </div>
          <div class="api-description">
            <p>Comprehensive date and time feature encoding with cyclical representations.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>SeasonLayer</h3>
          <div class="api-signature">
            <code>SeasonLayer()</code>
          </div>
          <div class="api-description">
            <p>Seasonal feature extraction from date/time data for temporal pattern recognition.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>CastToFloat32Layer</h3>
          <div class="api-signature">
            <code>CastToFloat32Layer()</code>
          </div>
          <div class="api-description">
            <p>Type casting utility layer for ensuring consistent data types throughout the model.</p>
          </div>
        </div>
      </div>
    </div>

    <div class="tab-content" id="feature-engineering-api">
      <div class="api-reference-grid">
        <div class="api-reference-card">
          <h3>VariableSelection</h3>
          <div class="api-signature">
            <code>VariableSelection(hidden_dim=64, dropout=0.1, use_context=False, context_dim=None)</code>
          </div>
          <div class="api-description">
            <p>Intelligent variable selection network for identifying important features using gated residual networks.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>GatedFeatureSelection</h3>
          <div class="api-signature">
            <code>GatedFeatureSelection(hidden_dim=64, dropout=0.1, activation='relu')</code>
          </div>
          <div class="api-description">
            <p>Learnable feature selection with gating network and residual connection for adaptive feature importance.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>GatedFeatureFusion</h3>
          <div class="api-signature">
            <code>GatedFeatureFusion(hidden_dim=128, dropout=0.1, activation='relu')</code>
          </div>
          <div class="api-description">
            <p>Gated mechanism for intelligently fusing multiple feature representations with learnable weights.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>AdvancedNumericalEmbedding</h3>
          <div class="api-signature">
            <code>AdvancedNumericalEmbedding(embedding_dim=64, num_bins=10, hidden_dim=128, dropout=0.1)</code>
          </div>
          <div class="api-description">
            <p>Advanced numerical feature embedding with dual-branch architecture for continuous and discrete features.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>DistributionAwareEncoder</h3>
          <div class="api-signature">
            <code>DistributionAwareEncoder(encoding_dim=64, dropout=0.1, detection_method='auto')</code>
          </div>
          <div class="api-description">
            <p>Distribution-aware feature encoding that automatically detects distribution type and applies appropriate transformations.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>DistributionTransformLayer</h3>
          <div class="api-signature">
            <code>DistributionTransformLayer(transform_type='auto', epsilon=1e-8, method='box-cox')</code>
          </div>
          <div class="api-description">
            <p>Automatic distribution transformation for numerical features to improve model performance.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>SparseAttentionWeighting</h3>
          <div class="api-signature">
            <code>SparseAttentionWeighting(temperature=1.0, dropout=0.1, sparsity_threshold=0.1)</code>
          </div>
          <div class="api-description">
            <p>Sparse attention weighting mechanisms for efficient computation and selective feature combination.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>FeatureCutout</h3>
          <div class="api-signature">
            <code>FeatureCutout(cutout_prob=0.1, noise_value=0.0, training_only=True)</code>
          </div>
          <div class="api-description">
            <p>Feature cutout for data augmentation and regularization in tabular data by randomly masking features.</p>
          </div>
        </div>
      </div>
    </div>

    <div class="tab-content" id="specialized-api">
      <div class="api-reference-grid">
        <div class="api-reference-card">
          <h3>GatedResidualNetwork</h3>
          <div class="api-signature">
            <code>GatedResidualNetwork(units, dropout_rate=0.2, name=None)</code>
          </div>
          <div class="api-description">
            <p>Gated residual network combining residual connections with gated linear units for improved gradient flow.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>GatedLinearUnit</h3>
          <div class="api-signature">
            <code>GatedLinearUnit(units, name=None)</code>
          </div>
          <div class="api-description">
            <p>Gated linear unit for intelligent feature gating and selective information flow control.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>TransformerBlock</h3>
          <div class="api-signature">
            <code>TransformerBlock(dim_model=32, num_heads=3, ff_units=16, dropout_rate=0.2)</code>
          </div>
          <div class="api-description">
            <p>Standard transformer block with multi-head attention and feed-forward networks for tabular data.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>TabularMoELayer</h3>
          <div class="api-signature">
            <code>TabularMoELayer(num_experts=4, expert_units=16, name=None)</code>
          </div>
          <div class="api-description">
            <p>Mixture of Experts for tabular data with adaptive expert selection and routing.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>BoostingBlock</h3>
          <div class="api-signature">
            <code>BoostingBlock(hidden_units=64, hidden_activation='relu', gamma_trainable=True, dropout_rate=None)</code>
          </div>
          <div class="api-description">
            <p>Gradient boosting inspired neural network block for sequential learning and residual correction.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>BoostingEnsembleLayer</h3>
          <div class="api-signature">
            <code>BoostingEnsembleLayer(num_learners=3, learner_units=64, hidden_activation='relu', dropout_rate=None)</code>
          </div>
          <div class="api-description">
            <p>Ensemble of boosting blocks for improved performance and robustness through parallel weak learners.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>BusinessRulesLayer</h3>
          <div class="api-signature">
            <code>BusinessRulesLayer(rules, feature_type, trainable_weights=True, weight_initializer='ones')</code>
          </div>
          <div class="api-description">
            <p>Integration of business rules and domain knowledge into neural networks for anomaly detection.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>StochasticDepth</h3>
          <div class="api-signature">
            <code>StochasticDepth(survival_prob=0.8, scale_at_test=True)</code>
          </div>
          <div class="api-description">
            <p>Stochastic depth regularization for improved training and generalization in deep networks.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>SlowNetwork</h3>
          <div class="api-signature">
            <code>SlowNetwork(hidden_units=64, num_layers=3, activation='relu', dropout=0.1)</code>
          </div>
          <div class="api-description">
            <p>Slow network architecture for careful and deliberate feature processing with controlled information flow.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>HyperZZWOperator</h3>
          <div class="api-signature">
            <code>HyperZZWOperator(hidden_units=64, hyperparameter_dim=32, activation='relu')</code>
          </div>
          <div class="api-description">
            <p>Hyperparameter-aware operator for adaptive model behavior and dynamic parameter adjustment.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>TextPreprocessingLayer</h3>
          <div class="api-signature">
            <code>TextPreprocessingLayer(max_length=100, vocab_size=10000, tokenizer='word')</code>
          </div>
          <div class="api-description">
            <p>Text preprocessing utilities for natural language features in tabular data with tokenization and encoding.</p>
          </div>
        </div>
      </div>
    </div>

    <div class="tab-content" id="utility-api">
      <div class="api-reference-grid">
        <div class="api-reference-card">
          <h3>CastToFloat32Layer</h3>
          <div class="api-signature">
            <code>CastToFloat32Layer(name=None)</code>
          </div>
          <div class="api-description">
            <p>Type casting utility layer for ensuring consistent data types throughout the model.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>AdvancedGraphFeature</h3>
          <div class="api-signature">
            <code>AdvancedGraphFeature(hidden_dim=64, num_heads=4, dropout=0.1, use_attention=True)</code>
          </div>
          <div class="api-description">
            <p>Advanced graph-based feature processing with dynamic adjacency learning and relationship modeling.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>GraphFeatureAggregation</h3>
          <div class="api-signature">
            <code>GraphFeatureAggregation(aggregation_method='mean', hidden_dim=64, dropout=0.1)</code>
          </div>
          <div class="api-description">
            <p>Graph feature aggregation mechanisms for relationship modeling and feature interaction learning.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>MultiHeadGraphFeaturePreprocessor</h3>
          <div class="api-signature">
            <code>MultiHeadGraphFeaturePreprocessor(num_heads=4, hidden_dim=64, dropout=0.1, aggregation='concat')</code>
          </div>
          <div class="api-description">
            <p>Multi-head graph feature preprocessing for complex feature interactions and relationship learning.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>NumericalAnomalyDetection</h3>
          <div class="api-signature">
            <code>NumericalAnomalyDetection(method='isolation_forest', contamination=0.1, threshold=0.5)</code>
          </div>
          <div class="api-description">
            <p>Anomaly detection for numerical features using statistical and machine learning methods.</p>
          </div>
        </div>

        <div class="api-reference-card">
          <h3>CategoricalAnomalyDetectionLayer</h3>
          <div class="api-signature">
            <code>CategoricalAnomalyDetectionLayer(method='frequency', threshold=0.01, min_frequency=5)</code>
          </div>
          <div class="api-description">
            <p>Anomaly detection for categorical features with pattern recognition and frequency analysis.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
