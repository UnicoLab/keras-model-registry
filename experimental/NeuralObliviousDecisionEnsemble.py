"""
A custom layer that mimics decision tree splits via learnable oblivious decision functions. NODE architectures have shown strong performance on tabular data by capturing non-linear interactions with fewer hyperparameters.
"""


import tensorflow as tf
from tensorflow.keras import layers
from loguru import logger

class NeuralObliviousDecisionEnsemble(layers.Layer):
    """Neural Oblivious Decision Ensemble (NODE) layer for tabular data.

    This layer implements an ensemble of oblivious decision trees in a differentiable way.
    For each tree, the decision function is shared across all nodes at the same level.
    Each tree of depth D has 2^D leaves; the probability of each leaf is computed as the
    product of soft decisions across levels. Each leaf has a learnable output vector, and
    the tree output is the weighted sum of the leaf outputs. An ensemble of trees is then
    averaged to produce the final output.

    Args:
        num_trees (int): Number of trees in the ensemble.
        tree_depth (int): Depth of each tree.
        output_dim (int): Dimensionality of the tree output (e.g. 1 for regression).

    Example:
        ```python
        import tensorflow as tf

        # Dummy data: 32 samples, 10 features.
        x = tf.random.normal((32, 10))
        node_layer = NeuralObliviousDecisionEnsemble(num_trees=5, tree_depth=3, output_dim=1)
        y = node_layer(x)
        print("NODE output shape:", y.shape)  # Expected shape: (32, 1)
        ```
    """
    def __init__(self, num_trees: int, tree_depth: int, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.output_dim = output_dim
        self.num_leaves = 2 ** tree_depth

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        # For each tree and each level, we learn a decision function:
        # weights: shape (num_trees, tree_depth, num_features)
        self.decision_weights = self.add_weight(
            name="decision_weights",
            shape=(self.num_trees, self.tree_depth, self.num_features),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Biases for decision functions: shape (num_trees, tree_depth)
        self.decision_biases = self.add_weight(
            name="decision_biases",
            shape=(self.num_trees, self.tree_depth),
            initializer="zeros",
            trainable=True,
        )
        # Each tree has num_leaves leaves with a learnable output vector of shape (output_dim,)
        self.leaf_values = self.add_weight(
            name="leaf_values",
            shape=(self.num_trees, self.num_leaves, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        logger.debug("NeuralObliviousDecisionEnsemble built: {} trees, depth {}, {} leaves per tree.",
                     self.num_trees, self.tree_depth, self.num_leaves)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Expand inputs for tree processing: shape (batch, 1, 1, num_features)
        batch_size = tf.shape(inputs)[0]
        x = tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)  # (batch, 1, 1, num_features)
        # Compute decision probabilities for each tree and each level.
        # Linear transformation: (batch, num_trees, tree_depth)
        # We use broadcasting to combine inputs with decision_weights and biases.
        decision_logits = tf.tensordot(inputs, self.decision_weights, axes=[[1], [2]])  # (batch, num_trees, tree_depth)
        decision_logits = decision_logits - self.decision_biases  # (batch, num_trees, tree_depth)
        # Apply sigmoid to get soft decisions.
        decisions = tf.sigmoid(decision_logits)  # (batch, num_trees, tree_depth)

        # For each tree, compute leaf probabilities.
        # We need to assign for each leaf (binary vector of length tree_depth) a probability:
        # p(leaf) = Π_{d=1}^{D} (decision_d if bit=1 else (1 - decision_d))
        # Create a binary mask for leaves: shape (num_leaves, tree_depth)
        bin_format = tf.reshape(tf.cast(tf.range(self.num_leaves), tf.int32), (-1, 1))
        # Generate binary representation with tree_depth bits.
        mask = tf.cast(tf.bitwise.right_shift(bin_format, tf.range(self.tree_depth)) & 1, tf.float32)  # (num_leaves, tree_depth)

        # Expand dimensions for broadcasting: decisions (batch, num_trees, 1, tree_depth), mask (1, 1, num_leaves, tree_depth)
        decisions_expanded = tf.expand_dims(decisions, axis=2)
        mask_expanded = tf.reshape(mask, (1, 1, self.num_leaves, self.tree_depth))
        # Compute probabilities: use mask: p = decision if mask==1, else (1-decision)
        leaf_probs = tf.math.pow(decisions_expanded, mask_expanded) * tf.math.pow(1 - decisions_expanded, 1 - mask_expanded)
        # Multiply probabilities along tree depth: (batch, num_trees, num_leaves)
        leaf_probs = tf.reduce_prod(leaf_probs, axis=-1)
        # For each tree, compute output as weighted sum of leaf values.
        # Multiply leaf probabilities (batch, num_trees, num_leaves) with leaf_values (num_trees, num_leaves, output_dim)
        # Use broadcasting: result: (batch, num_trees, output_dim)
        tree_outputs = tf.matmul(leaf_probs, self.leaf_values)  # (batch, num_trees, output_dim)
        # Average over trees.
        output = tf.reduce_mean(tree_outputs, axis=1)  # (batch, output_dim)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_trees": self.num_trees,
            "tree_depth": self.tree_depth,
            "output_dim": self.output_dim,
        })
        return config
