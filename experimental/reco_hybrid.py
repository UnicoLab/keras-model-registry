# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable
import tempfile
import os
from config import Config

# setting up the config file
CONF = Config()




@register_keras_serializable()
class CollaborativeFilteringModel(Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embedding = layers.Embedding(
            input_dim=num_users, 
            output_dim=embedding_dim, 
            embeddings_regularizer=l2(1e-6),
            name="user_embedding"
        )
        self.item_embedding = layers.Embedding(
            input_dim=num_items, 
            output_dim=embedding_dim, 
            embeddings_regularizer=l2(1e-6),
            name="item_embedding"
        )
        self.dropout = layers.Dropout(0.2)

    def call(self, inputs):
        user_ids, item_ids = inputs
        
        # Ensure input shapes
        user_ids = tf.ensure_shape(user_ids, [None])
        item_ids = tf.ensure_shape(item_ids, [None])
        
        # Get embeddings
        user_vecs = self.user_embedding(user_ids)  # Shape: (batch_size, embedding_dim)
        user_vecs = self.dropout(user_vecs)
        item_vecs = self.item_embedding(item_ids)  # Shape: (batch_size, embedding_dim)
        item_vecs = self.dropout(item_vecs)
        
        # Compute dot product
        dot_product = tf.reduce_sum(user_vecs * item_vecs, axis=1)  # Shape: (batch_size,)
        dot_product = tf.expand_dims(dot_product, -1)  # Shape: (batch_size, 1)
        
        return tf.ensure_shape(dot_product, [None, 1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class TwoTowerModel(Model):
    def __init__(self, user_dim, item_dim):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        
        # User tower
        self.user_tower = tf.keras.Sequential([
            layers.Dense(units=user_dim, activation='relu', kernel_regularizer=l2(1e-6)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
        ], name="user_tower")
        
        # Item tower
        self.item_tower = tf.keras.Sequential([
            layers.Dense(units=item_dim, activation='relu', kernel_regularizer=l2(1e-6)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
        ], name="item_tower")
        
        # Final dot product layer
        self.dot_product = layers.Dot(axes=1, normalize=True)
        self.final_reshape = layers.Reshape((1,))

    def call(self, user_features, item_features):
        # Process through towers
        user_embedding = self.user_tower(user_features)  # Shape: (batch_size, user_dim)
        item_embedding = self.item_tower(item_features)  # Shape: (batch_size, item_dim)
        
        # Ensure embeddings have the right shape
        user_embedding = tf.ensure_shape(user_embedding, [None, self.user_dim])
        item_embedding = tf.ensure_shape(item_embedding, [None, self.item_dim])
        
        # Compute similarity score
        similarity = self.dot_product([user_embedding, item_embedding])  # Shape: (batch_size,)
        similarity = self.final_reshape(similarity)  # Shape: (batch_size, 1)
        
        return tf.ensure_shape(similarity, [None, 1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "user_dim": self.user_dim,
            "item_dim": self.item_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class RankingModel(Model):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.dense = layers.Dense(units=feature_dim, activation='relu', kernel_regularizer=l2(1e-6))
        self.output_layer = layers.Dense(units=1, activation='linear')  # Linear activation to allow negative values
        self.dropout = layers.Dropout(0.3)

    def call(self, features):
        x = self.dense(features)
        x = self.dropout(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Unified Recommendation Model with Preprocessing
@register_keras_serializable()
class UnifiedRecommendationModelWithPreprocessing(Model):
    def __init__(self, num_users, num_items, embedding_dim, user_dim, item_dim, feature_dim, preprocessing_model_user, preprocessing_model_item):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.feature_dim = feature_dim
        self.preprocessing_model_user = preprocessing_model_user
        self.preprocessing_model_item = preprocessing_model_item

        # Submodels
        self.collab_model = CollaborativeFilteringModel(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)
        self.two_tower_model = TwoTowerModel(user_dim=user_dim, item_dim=item_dim)
        self.ranking_model = RankingModel(feature_dim=feature_dim)

        # Learnable weights for dynamic weighting
        self.collab_weight = tf.Variable(1.0, trainable=True, name="collab_weight")
        self.two_tower_weight = tf.Variable(1.0, trainable=True, name="two_tower_weight")
        self.ranking_weight = tf.Variable(1.0, trainable=True, name="ranking_weight")

    def build(self, input_shapes):
        # Define input shapes for each feature
        for feature_name, shape in input_shapes.items():
            if feature_name in CONF.data.FEATURES_SPECS_USER:
                expected_shape = (None, 1)  # (batch_size, 1)
                if len(shape) != 2 or shape[1] != 1:
                    raise ValueError(f"Expected shape (None, 1) for feature {feature_name}, got {shape}")
            elif feature_name in CONF.data.FEATURES_SPECS_ITEM:
                expected_shape = (None, 1)  # (batch_size, 1)
                if len(shape) != 2 or shape[1] != 1:
                    raise ValueError(f"Expected shape (None, 1) for feature {feature_name}, got {shape}")
        
        # Build preprocessing models with fixed output shapes
        sample_user_inputs = {k: tf.keras.layers.Input(shape=(1,), name=k) 
                            for k in CONF.data.FEATURES_SPECS_USER.keys()}
        sample_item_inputs = {k: tf.keras.layers.Input(shape=(1,), name=k) 
                            for k in CONF.data.FEATURES_SPECS_ITEM.keys()}
        
        # Get output shapes from preprocessing models
        user_features = self.preprocessing_model_user(sample_user_inputs)
        item_features = self.preprocessing_model_item(sample_item_inputs)
        
        # Set output shapes for preprocessing models
        self.user_feature_shape = user_features.shape[1:]  # Remove batch dimension
        self.item_feature_shape = item_features.shape[1:]  # Remove batch dimension
        
        print(f"\n[DEBUG] Set preprocessing output shapes:")
        print(f"User features shape: {self.user_feature_shape}")
        print(f"Item features shape: {self.item_feature_shape}")
        
        super().build(input_shapes)

    def call(self, inputs):
        # Debug input data
        print("\n=== INPUT DATA DEBUG ===")
        print("Raw inputs:")
        for k, v in inputs.items():
            print(f"  {k}:")
            print(f"    Shape: {v.shape}")
            print(f"    Dtype: {v.dtype}")
            try:
                # Only try to show values if tensor is concrete (not symbolic)
                if hasattr(v, 'numpy'):
                    print(f"    First 3 values: {v[0:3].numpy()}")
                    if v.dtype in (tf.float32, tf.float64, tf.int32, tf.int64):
                        print(f"    Min/Max: {tf.reduce_min(v).numpy()}/{tf.reduce_max(v).numpy()}")
            except Exception:
                pass

        # Split inputs into user and item features based on config
        user_inputs = {
            k: v for k, v in inputs.items() 
            if k in list(CONF.data.FEATURES_SPECS_USER.keys())
        }
        item_inputs = {
            k: v for k, v in inputs.items() 
            if k in list(CONF.data.FEATURES_SPECS_ITEM.keys())
        }
        
        print("\n=== SPLIT FEATURES DEBUG ===")
        print("User features:")
        for k, v in user_inputs.items():
            print(f"  {k}:")
            print(f"    Shape: {v.shape}")
            print(f"    Spec: {CONF.data.FEATURES_SPECS_USER.get(k)}")
            try:
                if hasattr(v, 'numpy'):
                    print(f"    First 3 values: {v[0:3].numpy()}")
                    if v.dtype in (tf.float32, tf.float64, tf.int32, tf.int64):
                        print(f"    Min/Max: {tf.reduce_min(v).numpy()}/{tf.reduce_max(v).numpy()}")
            except Exception:
                pass
        
        print("\nItem features:")
        for k, v in item_inputs.items():
            print(f"  {k}:")
            print(f"    Shape: {v.shape}")
            print(f"    Spec: {CONF.data.FEATURES_SPECS_ITEM.get(k)}")
            try:
                if hasattr(v, 'numpy'):
                    print(f"    First 3 values: {v[0:3].numpy()}")
                    if v.dtype in (tf.float32, tf.float64, tf.int32, tf.int64):
                        print(f"    Min/Max: {tf.reduce_min(v).numpy()}/{tf.reduce_max(v).numpy()}")
            except Exception:
                pass
        
        print("\n=== PREPROCESSING DEBUG ===")
        print("Before preprocessing:")
        print(f"User inputs shapes:")
        for k, v in user_inputs.items():
            print(f"  {k}: {tf.shape(v)}")
        
        # Add shape debugging for calendar_date specifically
        if 'calendar_date' in user_inputs:
            print("\nCalendar date debug:")
            print(f"Original shape: {tf.shape(user_inputs['calendar_date'])}")
            print(f"Dtype: {user_inputs['calendar_date'].dtype}")
            # Print the actual tensor
            print(f"Tensor: {user_inputs['calendar_date']}")
            
        user_features = self.preprocessing_model_user(user_inputs)
        item_features = self.preprocessing_model_item(item_inputs)
        
        print("\n=== PREPROCESSED FEATURES DEBUG ===")
        print(f"User features:")
        print(f"  Shape: {user_features.shape}")
        try:
            if hasattr(user_features, 'numpy'):
                print(f"  First 3 rows:\n{user_features[0:3].numpy()}")
                print(f"  Min/Max: {tf.reduce_min(user_features).numpy()}/{tf.reduce_max(user_features).numpy()}")
        except Exception:
            pass
        
        print(f"\nItem features:")
        print(f"  Shape: {item_features.shape}")
        try:
            if hasattr(item_features, 'numpy'):
                print(f"  First 3 rows:\n{item_features[0:3].numpy()}")
                print(f"  Min/Max: {tf.reduce_min(item_features).numpy()}/{tf.reduce_max(item_features).numpy()}")
        except Exception:
            pass
        
        # Shape enforcement
        user_features = tf.ensure_shape(user_features, (None,) + self.user_feature_shape)
        item_features = tf.ensure_shape(item_features, (None,) + self.item_feature_shape)
        
        # Extract user and item IDs for collaborative filtering
        user_ids = tf.squeeze(tf.cast(inputs['entity_number'], tf.int32))
        item_ids = tf.squeeze(tf.cast(inputs['product_number'], tf.int32))
        
        # Ensure ID shapes
        user_ids = tf.ensure_shape(user_ids, [None])
        item_ids = tf.ensure_shape(item_ids, [None])
        
        # Collaborative Filtering scores - Shape: (batch_size, 1)
        collab_scores = self.collab_model([user_ids, item_ids])
        collab_scores = tf.ensure_shape(collab_scores, [None, 1])
        print(f"\n[DEBUG] Collab scores shape: {collab_scores.shape}")
        
        # Two-Tower Model scores - Shape: (batch_size, 1)
        retrieval_scores = self.two_tower_model(user_features, item_features)
        retrieval_scores = tf.ensure_shape(retrieval_scores, [None, 1])
        print(f"[DEBUG] Retrieval scores shape: {retrieval_scores.shape}")
        
        # Combine features for ranking - Shape: (batch_size, user_dim + item_dim + 1)
        # Cast all tensors to float32 before concatenation
        user_features = tf.cast(user_features, tf.float32)
        item_features = tf.cast(item_features, tf.float32)
        collab_scores = tf.cast(collab_scores, tf.float32)
        
        ranking_input = tf.concat([
            user_features,    # Shape: (batch_size, user_feature_dim)
            item_features,    # Shape: (batch_size, item_feature_dim)
            collab_scores     # Shape: (batch_size, 1)
        ], axis=1)
        ranking_input = tf.ensure_shape(ranking_input, [None, self.user_feature_shape[0] + self.item_feature_shape[0] + 1])
        print(f"[DEBUG] Ranking input shape: {ranking_input.shape}")
        
        # Get quantity prediction from ranking model - Shape: (batch_size, 1)
        quantity_prediction = self.ranking_model(ranking_input)
        quantity_prediction = tf.ensure_shape(quantity_prediction, [None, 1])
        print(f"[DEBUG] Quantity prediction shape: {quantity_prediction.shape}")
        
        # Dynamic weighting of scores - All shapes: (batch_size, 1)
        weighted_collab = self.collab_weight * collab_scores
        weighted_retrieval = self.two_tower_weight * retrieval_scores
        weighted_ranking = self.ranking_weight * quantity_prediction
        
        # Combine scores - Shape: (batch_size, 1)
        combined_scores = weighted_collab + weighted_retrieval + weighted_ranking
        combined_scores = tf.ensure_shape(combined_scores, [None, 1])
        print(f"\n[DEBUG] Combined scores shape: {combined_scores.shape}")
        
        print("\n=== COLLABORATIVE FILTERING DEBUG ===")
        print(f"Collab scores:")
        print(f"  Shape: {collab_scores.shape}")
        try:
            if hasattr(collab_scores, 'numpy'):
                print(f"  First 3 scores: {collab_scores[0:3].numpy()}")
                print(f"  Min/Max: {tf.reduce_min(collab_scores).numpy()}/{tf.reduce_max(collab_scores).numpy()}")
        except Exception:
            pass
        
        print("\n=== RETRIEVAL SCORES DEBUG ===")
        print(f"Retrieval scores:")
        print(f"  Shape: {retrieval_scores.shape}")
        try:
            if hasattr(retrieval_scores, 'numpy'):
                print(f"  First 3 scores: {retrieval_scores[0:3].numpy()}")
                print(f"  Min/Max: {tf.reduce_min(retrieval_scores).numpy()}/{tf.reduce_max(retrieval_scores).numpy()}")
        except Exception:
            pass
        
        print("\n=== COMBINED FEATURES DEBUG ===")
        print(f"Combined features:")
        print(f"  Shape: {ranking_input.shape}")
        try:
            if hasattr(ranking_input, 'numpy'):
                print(f"  First 3 rows:\n{ranking_input[0:3].numpy()}")
                print(f"  Min/Max: {tf.reduce_min(ranking_input).numpy()}/{tf.reduce_max(ranking_input).numpy()}")
        except Exception:
            pass
        
        print("\n=== FINAL OUTPUT DEBUG ===")
        print(f"Quantity prediction:")
        print(f"  Shape: {quantity_prediction.shape}")
        try:
            if hasattr(quantity_prediction, 'numpy'):
                print(f"  First 3 predictions: {quantity_prediction[0:3].numpy()}")
                print(f"  Min/Max: {tf.reduce_min(quantity_prediction).numpy()}/{tf.reduce_max(quantity_prediction).numpy()}")
        except Exception:
            pass
        
        try:
            print("\n=== COLLABORATIVE FILTERING DEBUG ===")
            print(f"Collab scores:")
            print(f"  Shape: {collab_scores.shape}")
            try:
                if hasattr(collab_scores, 'numpy'):
                    print(f"  First 3 scores: {collab_scores[0:3].numpy()}")
                    print(f"  Min/Max: {tf.reduce_min(collab_scores).numpy()}/{tf.reduce_max(collab_scores).numpy()}")
            except Exception:
                pass
            
            print("\n=== RETRIEVAL SCORES DEBUG ===")
            print(f"Retrieval scores:")
            print(f"  Shape: {retrieval_scores.shape}")
            try:
                if hasattr(retrieval_scores, 'numpy'):
                    print(f"  First 3 scores: {retrieval_scores[0:3].numpy()}")
                    print(f"  Min/Max: {tf.reduce_min(retrieval_scores).numpy()}/{tf.reduce_max(retrieval_scores).numpy()}")
            except Exception:
                pass
            
            print("\n=== COMBINED FEATURES DEBUG ===")
            print(f"Combined features:")
            print(f"  Shape: {ranking_input.shape}")
            try:
                if hasattr(ranking_input, 'numpy'):
                    print(f"  First 3 rows:\n{ranking_input[0:3].numpy()}")
                    print(f"  Min/Max: {tf.reduce_min(ranking_input).numpy()}/{tf.reduce_max(ranking_input).numpy()}")
            except Exception:
                pass
            
            print("\n=== FINAL OUTPUT DEBUG ===")
            print(f"Quantity prediction:")
            print(f"  Shape: {quantity_prediction.shape}")
            try:
                if hasattr(quantity_prediction, 'numpy'):
                    print(f"  First 3 predictions: {quantity_prediction[0:3].numpy()}")
                    print(f"  Min/Max: {tf.reduce_min(quantity_prediction).numpy()}/{tf.reduce_max(quantity_prediction).numpy()}")
            except Exception:
                pass
        except Exception as e:
            print(f"Error occurred during debug logging: {str(e)}")
        
        return {
            "sale_quantity": quantity_prediction,
            "sale_amount_vat_included": combined_scores
        }

    def get_config(self):
        print("\n[DEBUG] UnifiedRecommendationModelWithPreprocessing.get_config")
        config = super().get_config()
        print("Base config keys:", list(config.keys()))
        
        # Convert preprocessing models to config
        preprocessing_user_config = self.preprocessing_model_user.get_config()
        preprocessing_item_config = self.preprocessing_model_item.get_config()
        
        config.update({
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "user_dim": self.user_dim,
            "item_dim": self.item_dim,
            "feature_dim": self.feature_dim,
            "preprocessing_model_user": preprocessing_user_config,
            "preprocessing_model_item": preprocessing_item_config,
            "collab_weight": float(self.collab_weight.numpy()),
            "two_tower_weight": float(self.two_tower_weight.numpy()),
            "ranking_weight": float(self.ranking_weight.numpy()),
        })
        print("Updated config keys:", list(config.keys()))
        return config

    @classmethod
    def from_config(cls, config):
        print("\n[DEBUG] UnifiedRecommendationModelWithPreprocessing.from_config")
        print("Received config keys:", list(config.keys()))
        
        # Extract preprocessing model configs
        preprocessing_user_config = config.pop("preprocessing_model_user", {})
        preprocessing_item_config = config.pop("preprocessing_model_item", {})
        print("Successfully extracted preprocessing_model configs")

        # Register custom objects
        custom_objects = {
            'StringLookup': tf.keras.layers.StringLookup,
            'Normalization': tf.keras.layers.Normalization,
            'Rescaling': tf.keras.layers.Rescaling,
        }

        # Recreate preprocessing models from config
        preprocessing_model_user = tf.keras.Model.from_config(
            preprocessing_user_config,
            custom_objects=custom_objects
        )
        preprocessing_model_item = tf.keras.Model.from_config(
            preprocessing_item_config,
            custom_objects=custom_objects
        )
        
        # Create instance with the preprocessing models
        instance = cls(
            num_users=config["num_users"],
            num_items=config["num_items"],
            embedding_dim=config["embedding_dim"],
            user_dim=config["user_dim"],
            item_dim=config["item_dim"],
            feature_dim=config["feature_dim"],
            preprocessing_model_user=preprocessing_model_user,
            preprocessing_model_item=preprocessing_model_item,
        )

        # Set weights
        instance.collab_weight.assign(config["collab_weight"])
        instance.two_tower_weight.assign(config["two_tower_weight"])
        instance.ranking_weight.assign(config["ranking_weight"])

        return instance


# Feedback Layer for adjustments
@register_keras_serializable()
class FeedbackLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, feedback):
        if feedback is not None:
            return inputs * feedback
        return inputs

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Similarity Layer for explainability
@register_keras_serializable()
class SimilarityLayer(layers.Layer):
    def __init__(self, user_embedding_layer, **kwargs):
        super().__init__(**kwargs)
        self.user_embedding_layer = user_embedding_layer

    def get_user_embeddings(self):
        return self.user_embedding_layer.embeddings

    def cosine_similarity(self, vec1, vec2):
        dot_product = tf.reduce_sum(vec1 * vec2, axis=-1)
        norm1 = tf.norm(vec1, axis=-1)
        norm2 = tf.norm(vec2, axis=-1)
        return dot_product / (norm1 * norm2)

    def get_top_similar_users(self, user_id, top_n=5):
        target_embedding = self.user_embedding_layer(user_id)
        all_embeddings = self.get_user_embeddings()
        
        similarities = self.cosine_similarity(
            tf.expand_dims(target_embedding, 0),
            all_embeddings
        )
        
        top_k_values, top_k_indices = tf.nn.top_k(similarities, k=top_n + 1)
        return top_k_values[1:], top_k_indices[1:]  # Exclude self-similarity

    def get_config(self):
        config = super().get_config()
        config.update({
            "user_embedding_layer": tf.keras.layers.serialize(self.user_embedding_layer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        user_embedding_layer = tf.keras.layers.deserialize(config.pop("user_embedding_layer"))
        return cls(user_embedding_layer=user_embedding_layer, **config)


# Explainable Unified Model
@register_keras_serializable()
class ExplainableUnifiedModelWithPreprocessing(UnifiedRecommendationModelWithPreprocessing):
    def __init__(self, num_users, num_items, embedding_dim, user_dim, item_dim, feature_dim, preprocessing_model_user, preprocessing_model_item):
        super().__init__(num_users, num_items, embedding_dim, user_dim, item_dim, feature_dim, preprocessing_model_user, preprocessing_model_item)
        self.similarity_layer = SimilarityLayer(self.collab_model.user_embedding)
        self.feedback_layer = FeedbackLayer()

    def call(self, inputs, feedback=None):
        # Get base model predictions
        base_outputs = super().call(inputs)
        
        # Apply feedback if provided
        if feedback is not None:
            base_outputs = self.feedback_layer(base_outputs, feedback)
        
        return base_outputs

    def get_top_similar_users(self, user_id, top_n=5):
        return self.similarity_layer.get_top_similar_users(user_id, top_n)

    def get_config(self):
        print("\n[DEBUG] ExplainableUnifiedModelWithPreprocessing.get_config")
        config = super().get_config()
        print("Base config keys:", list(config.keys()))
        
        # Add custom layer configs
        config.update({
            "similarity_layer": tf.keras.layers.serialize(self.similarity_layer),
            "feedback_layer": tf.keras.layers.serialize(self.feedback_layer)
        })
        print("Updated config keys:", list(config.keys()))
        return config

    @classmethod
    def from_config(cls, config):
        print("\n[DEBUG] ExplainableUnifiedModelWithPreprocessing.from_config")
        print("Received config keys:", list(config.keys()))

        # Extract custom layer configs
        similarity_config = config.pop("similarity_layer", None)
        feedback_config = config.pop("feedback_layer", None)
        print("Successfully extracted custom layer configs")

        # Create base instance
        instance = super().from_config(config)
        print("Successfully created base instance")

        # Deserialize and set custom layers
        if similarity_config:
            instance.similarity_layer = tf.keras.layers.deserialize(similarity_config)
        if feedback_config:
            instance.feedback_layer = tf.keras.layers.deserialize(feedback_config)
        print("Successfully set custom layers")

        return instance
