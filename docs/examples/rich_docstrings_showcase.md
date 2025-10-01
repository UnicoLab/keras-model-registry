# ✨ Rich Docstrings Showcase

This page demonstrates the comprehensive documentation that KMR provides through its rich docstrings. Each layer and model includes detailed documentation with examples, best practices, and implementation guidance.

## 🧠 AdvancedGraphFeatureLayer

The `AdvancedGraphFeatureLayer` is an excellent example of comprehensive documentation. It includes:

### Complete Parameter Documentation
- **embed_dim**: Dimensionality of the projected feature embeddings
- **num_heads**: Number of attention heads with validation
- **dropout_rate**: Dropout rate for regularization
- **hierarchical**: Whether to apply hierarchical aggregation
- **num_groups**: Number of groups for clustering

### Detailed Usage Examples

The layer provides three complete examples:

1. **Basic Usage**: Simple tabular data processing
2. **With Hierarchical Aggregation**: Advanced feature grouping
3. **Without Training**: Inference mode usage

### Best Practices and Performance Notes

The documentation includes:
- **When to Use**: Specific scenarios where the layer excels
- **Best Practices**: Recommended parameter values and usage patterns
- **Performance Considerations**: Memory usage and scalability notes

### Implementation Details

- Complete method documentation with parameter types
- Input/output shape specifications
- Error handling and validation
- Keras 3 compatibility notes

## TabularAttention

Another excellent example with comprehensive documentation:

### Dual Attention Mechanism
- Inter-feature attention for feature dependencies
- Inter-sample attention for sample relationships
- Multi-head attention implementation

### Complete API Documentation
- Parameter validation and type checking
- Input/output shape specifications
- Training vs inference mode handling

### Practical Examples
- Real-world usage with sample data
- Shape transformations and projections
- Integration with other Keras layers

## AdvancedNumericalEmbedding

This layer showcases advanced documentation patterns:

### Dual Branch Architecture
- Continuous branch with MLP processing
- Discrete branch with learnable binning
- Learnable gate for branch combination

### Comprehensive Parameter Guide
- All parameters with types and defaults
- Validation logic and error messages
- Performance optimization tips

### Implementation Architecture
- Detailed build process explanation
- Branch construction methodology
- Output shape computation

## Documentation Standards

All KMR layers follow consistent documentation standards:

### Required Elements
- **Class docstring**: Complete description with architecture overview
- **Parameter documentation**: Types, defaults, and validation rules
- **Usage examples**: Multiple scenarios with code samples
- **Best practices**: Performance and usage recommendations
- **Implementation notes**: Technical details for developers

### Code Examples
- **Basic usage**: Simple, clear examples
- **Advanced usage**: Complex scenarios with explanations
- **Integration examples**: How to combine with other layers
- **Error handling**: Common mistakes and solutions

### Type Annotations
- Complete type hints for all parameters
- Return type specifications
- Input/output shape documentation
- Keras 3 compatibility notes

## Benefits of Rich Documentation

The comprehensive docstrings provide:

1. **Developer Experience**: Clear understanding of layer capabilities
2. **Best Practices**: Guidance on optimal usage patterns
3. **Performance Insights**: Memory and computational considerations
4. **Integration Help**: How to combine layers effectively
5. **Error Prevention**: Validation rules and common pitfalls

## Automatic Documentation Generation

The mkdocstrings integration automatically generates beautiful documentation from these rich docstrings, including:

- **Syntax highlighting** for code examples
- **Cross-references** between related components
- **Search functionality** across all documentation
- **Responsive design** for all devices
- **Interactive examples** with copy-paste functionality

This approach ensures that the documentation stays up-to-date with the code and provides developers with all the information they need to use KMR effectively.
