import tensorflow as tf
from tensorflow.keras.layers import Layer


class PreprocessingLayer(Layer):
    """
    A custom layer for preprocessing data, including filtering, date handling, and feature engineering.
    """

    def __init__(self, min_sales=10, min_stores=5, **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
        self.min_sales = min_sales
        self.min_stores = min_stores

    def call(self, inputs):
        # Convert degrees to radians
        latitude = inputs["latitude"] * tf.constant(
            2 * 3.14159 / 360.0,
        )  # shape: (batch_size,)
        longitude = inputs["longitude"] * tf.constant(
            2 * 3.14159 / 360.0,
        )  # shape: (batch_size,)
        sales = inputs["sale_quantity"]  # shape: (batch_size,)

        # Create a mask for filtering without expanding dimensions
        mask = tf.cast(
            tf.greater(sales, self.min_sales),
            dtype=tf.float32,
        )  # shape: (batch_size,)

        # Apply mask by multiplication (no need for squeeze/expand_dims)
        masked_lat = latitude * mask  # shape: (batch_size,)
        masked_lon = longitude * mask  # shape: (batch_size,)
        masked_sales = sales * mask  # shape: (batch_size,)

        return {
            "latitude_rad": masked_lat,
            "longitude_rad": masked_lon,
            "sales": masked_sales,
            "mask": mask,
        }

    def compute_output_shape(self, input_shape):
        batch_size = input_shape["latitude"][0]
        return {
            "latitude_rad": (batch_size,),
            "longitude_rad": (batch_size,),
            "sales": (batch_size,),
            "mask": (batch_size,),
        }


class DistanceLayer(Layer):
    """
    A custom layer for calculating distances using the haversine formula.
    """

    def __init__(self, **kwargs):
        super(DistanceLayer, self).__init__(**kwargs)
        self.earth_radius = 6367.445  # Earth's radius in km

    def call(self, inputs):
        lat1, lon1, lat2, lon2 = inputs

        # Reshape inputs to ensure proper broadcasting
        lat1 = tf.reshape(lat1, [-1, 1])  # [batch_size, 1]
        lon1 = tf.reshape(lon1, [-1, 1])  # [batch_size, 1]
        lat2 = tf.reshape(lat2, [-1])  # [batch_size]
        lon2 = tf.reshape(lon2, [-1])  # [batch_size]

        # Calculate haversine formula components
        delta_lat = tf.expand_dims(lat2, 1) - lat1
        delta_lon = tf.expand_dims(lon2, 1) - lon1

        # Calculate using broadcasting
        a = (
            tf.sin(delta_lat / 2) ** 2
            + tf.cos(lat1)
            * tf.cos(tf.expand_dims(lat2, 1))
            * tf.sin(delta_lon / 2) ** 2
        )

        c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
        distances = self.earth_radius * c  # [batch_size, batch_size]

        # Normalize distances to [0, 1] range for better training
        max_dist = tf.reduce_max(distances)
        min_dist = tf.reduce_min(distances)
        normalized_distances = (distances - min_dist) / (max_dist - min_dist + 1e-6)

        return normalized_distances

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, batch_size)


class ClusteringLayer(Layer):
    """
    A custom clustering layer for grouping stores or products based on distances.
    """

    def __init__(self, n_clusters=5, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.batch_norm = None  # Will be initialized in build()

    def build(self, input_shape):
        # Initialize cluster centroids for distance-based clustering
        self.cluster_weights = self.add_weight(
            name="cluster_weights",
            shape=(self.n_clusters, self.n_clusters),
            initializer="random_normal",
            trainable=True,
        )

        # Initialize batch normalization with correct shape
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_norm.build((input_shape[0], self.n_clusters))

        self.built = True

    def call(self, inputs, training=None):
        # inputs shape: [batch_size, batch_size] (distance matrix)
        batch_size = tf.shape(inputs)[0]

        # Extract features from distance matrix
        features = tf.reduce_mean(inputs, axis=1, keepdims=True)  # [batch_size, 1]
        features = tf.tile(features, [1, 3])  # [batch_size, 3]

        # Project to cluster space
        cluster_logits = tf.matmul(
            features,
            self.cluster_weights[:3, :],
        )  # [batch_size, n_clusters]

        # Apply batch normalization
        normalized = self.batch_norm(cluster_logits, training=training)

        # Convert to probabilities
        cluster_probs = tf.nn.softmax(normalized, axis=-1)
        return cluster_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)


class RankingLayer(Layer):
    """
    A custom ranking layer that scores products for each store.
    """

    def __init__(self, embedding_dim=32, input_dim=5, **kwargs):
        super(RankingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.dense1 = tf.keras.layers.Dense(embedding_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.batch_norm1 = None  # Will be initialized in build()
        self.batch_norm2 = None  # Will be initialized in build()

    def build(self, input_shape):
        # Initialize batch normalization with correct shapes
        self.batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_norm2 = tf.keras.layers.BatchNormalization(axis=-1)

        # Build dense layers
        self.dense1.build(input_shape)
        dense1_output_shape = (input_shape[0], self.embedding_dim)
        self.dense2.build(dense1_output_shape)

        # Build batch norm layers
        self.batch_norm1.build(dense1_output_shape)
        self.batch_norm2.build((input_shape[0], 1))

        self.built = True

    def call(self, inputs, training=None):
        # First dense layer and batch norm
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)

        # Second dense layer and batch norm
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)

        # Calculate similarity between all pairs
        similarity = tf.matmul(x, tf.transpose(x))  # [batch_size, batch_size]

        # Scale similarity scores
        scaled_similarity = similarity / tf.sqrt(
            tf.cast(self.embedding_dim, tf.float32),
        )

        # Convert to probabilities using sigmoid
        scores = tf.sigmoid(scaled_similarity)

        return scores

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, batch_size)


class BatchIndexLayer(Layer):
    """
    A custom layer that generates batch indices.
    """

    def call(self, inputs):
        # Use tf.shape for dynamic batch size
        batch_size = tf.shape(inputs)[0]
        return tf.cast(tf.range(batch_size), dtype=inputs.dtype)

    def compute_output_shape(self, input_shape):
        return (None,)  # Dynamic batch size


class TopKLayer(Layer):
    """
    A custom layer to select the top-K products for each store.
    """

    def __init__(self, k=10, **kwargs):
        super(TopKLayer, self).__init__(**kwargs)
        self.k = k

    def call(self, scores):
        # Ensure k is not larger than the available scores
        actual_k = tf.minimum(self.k, tf.shape(scores)[-1])
        top_k_scores, top_k_indices = tf.math.top_k(scores, k=actual_k)
        return top_k_indices, top_k_scores

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return [(batch_size, self.k), (batch_size, self.k)]


class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


class WiserModel(tf.keras.Model):
    """
    A custom model class that implements the Wiser recommendation system architecture.
    """

    def __init__(self, n_clusters=5, embedding_dim=32, top_k=10, **kwargs):
        super(WiserModel, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.top_k = top_k

        # Initialize layers
        self.preprocessing = PreprocessingLayer()
        self.distance = DistanceLayer()
        self.clustering = ClusteringLayer(n_clusters=n_clusters)
        self.ranking = RankingLayer(embedding_dim=embedding_dim, input_dim=n_clusters)
        self.topk = TopKLayer(k=top_k)

    def call(self, inputs, training=None):
        # Preprocessing
        preprocessed = self.preprocessing(inputs)

        # Calculate pairwise distances between all points
        distances = self.distance(
            [
                preprocessed["latitude_rad"],
                preprocessed["longitude_rad"],
                preprocessed["latitude_rad"],  # Using same points for self-distances
                preprocessed["longitude_rad"],
            ],
        )

        # Cluster the points based on distances
        clusters = self.clustering(distances)

        # Generate rankings based on cluster assignments
        scores = self.ranking(clusters)

        # Select top K results
        indices, top_scores = self.topk(scores)

        return {"indices": indices, "scores": top_scores}

    def train_step(self, data):
        x = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(x, training=True)

            # Calculate losses
            loss_dict = {}

            # For indices (encourage diversity in cluster assignments)
            if "indices" in outputs:
                cluster_probs = tf.cast(outputs["indices"], tf.float32)

                # 1. Entropy loss to encourage diverse cluster assignments
                safe_probs = tf.clip_by_value(cluster_probs, 1e-10, 1.0)
                entropy = -tf.reduce_mean(
                    tf.reduce_sum(cluster_probs * tf.math.log(safe_probs), axis=1),
                )

                # 2. Encourage confident assignments (minimize entropy per sample)
                confidence = -tf.reduce_mean(
                    tf.reduce_sum(safe_probs * tf.math.log(safe_probs), axis=1),
                )

                # Adjusted weights for better balance
                loss_dict["indices"] = -2.0 * entropy + 0.5 * confidence

            # For scores (encourage meaningful rankings)
            if "scores" in outputs:
                scores = tf.cast(outputs["scores"], tf.float32)

                # 1. Encourage variation in scores with stronger weight
                score_variance = tf.math.reduce_variance(scores, axis=1)
                mean_variance = tf.reduce_mean(score_variance)

                # 2. Encourage scores to use full range (0 to 1)
                score_mean = tf.reduce_mean(scores)
                range_loss = tf.abs(score_mean - 0.5)

                # 3. Encourage sparsity (not all high scores) with reduced weight
                sparsity = tf.reduce_mean(tf.reduce_sum(scores, axis=1))

                # Adjusted weights for better balance
                loss_dict["scores"] = (
                    0.5 * range_loss - 0.3 * mean_variance + 0.005 * sparsity
                )

            # Compute total loss with L2 regularization
            l2_loss = (
                tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables]) * 0.001
            )
            total_loss = tf.add_n(list(loss_dict.values())) + l2_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars, strict=False))

        # Return metrics
        metrics = {**loss_dict, "total_loss": total_loss, "l2_loss": l2_loss}
        return metrics
