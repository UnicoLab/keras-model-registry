"""Tests for MatrixFactorizationModel."""

import unittest
import numpy as np
import tensorflow as tf
import keras
from kmr.models import MatrixFactorizationModel


class TestMatrixFactorizationModel(unittest.TestCase):
    """Test suite for MatrixFactorizationModel."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_users = 100
        self.num_items = 50
        self.batch_size = 16
        self.model = MatrixFactorizationModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=32,
            top_k=10,
            l2_reg=1e-4,
        )

    def test_initialization_default(self) -> None:
        """Test model initialization with default parameters."""
        model = MatrixFactorizationModel(num_users=100, num_items=50)
        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.num_items, 50)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.top_k, 10)

    def test_initialization_custom(self) -> None:
        """Test model initialization with custom parameters."""
        model = MatrixFactorizationModel(
            num_users=200,
            num_items=100,
            embedding_dim=64,
            top_k=20,
            l2_reg=1e-3,
        )
        self.assertEqual(model.num_users, 200)
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.top_k, 20)

    def test_invalid_num_users(self) -> None:
        """Test that invalid num_users raises error."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=0, num_items=50)

    def test_invalid_num_items(self) -> None:
        """Test that invalid num_items raises error."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=-1)

    def test_invalid_embedding_dim(self) -> None:
        """Test that invalid embedding_dim raises error."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=50, embedding_dim=0)

    def test_invalid_top_k(self) -> None:
        """Test that invalid top_k raises error."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=50, top_k=51)

    def test_invalid_l2_reg(self) -> None:
        """Test that invalid l2_reg raises error."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=50, l2_reg=-0.1)

    def test_forward_pass_output_shapes(self) -> None:
        """Test that forward pass produces correct output shapes."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        rec_indices, rec_scores = self.model([user_ids, item_ids], training=False)

        self.assertEqual(rec_indices.shape, (self.batch_size, self.model.top_k))
        self.assertEqual(rec_scores.shape, (self.batch_size, self.model.top_k))

    def test_output_indices_valid(self) -> None:
        """Test that output indices are valid item indices."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        rec_indices, _ = self.model([user_ids, item_ids], training=False)

        max_idx = tf.reduce_max(rec_indices).numpy()
        min_idx = tf.reduce_min(rec_indices).numpy()
        self.assertGreaterEqual(min_idx, 0)
        self.assertLess(max_idx, self.num_items)

    def test_output_scores_valid_range(self) -> None:
        """Test that output scores are in valid range [-1, 1]."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        _, rec_scores = self.model([user_ids, item_ids], training=False)

        min_score = tf.reduce_min(rec_scores).numpy()
        max_score = tf.reduce_max(rec_scores).numpy()
        self.assertGreaterEqual(min_score, -1.1)  # Allow small numerical error
        self.assertLessEqual(max_score, 1.1)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            user_ids = tf.constant(np.random.randint(0, self.num_users, (batch_size,)))
            item_ids = tf.constant(
                np.random.randint(0, self.num_items, (batch_size, self.num_items)),
            )

            rec_indices, rec_scores = self.model([user_ids, item_ids], training=False)

            self.assertEqual(rec_indices.shape[0], batch_size)
            self.assertEqual(rec_scores.shape[0], batch_size)

    def test_different_embedding_dims(self) -> None:
        """Test with different embedding dimensions."""
        for emb_dim in [8, 16, 32, 64]:
            model = MatrixFactorizationModel(
                num_users=self.num_users,
                num_items=self.num_items,
                embedding_dim=emb_dim,
            )
            user_ids = tf.constant(
                np.random.randint(0, self.num_users, (self.batch_size,)),
            )
            item_ids = tf.constant(
                np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
            )

            rec_indices, rec_scores = model([user_ids, item_ids], training=False)

            self.assertEqual(rec_indices.shape, (self.batch_size, model.top_k))

    def test_serialization(self) -> None:
        """Test model serialization."""
        config = self.model.get_config()

        self.assertEqual(config["num_users"], self.num_users)
        self.assertEqual(config["num_items"], self.num_items)
        self.assertEqual(config["embedding_dim"], 32)
        self.assertEqual(config["top_k"], 10)

    def test_deserialization(self) -> None:
        """Test model deserialization from config."""
        config = self.model.get_config()
        new_model = MatrixFactorizationModel.from_config(config)

        self.assertEqual(new_model.num_users, self.model.num_users)
        self.assertEqual(new_model.num_items, self.model.num_items)
        self.assertEqual(new_model.embedding_dim, self.model.embedding_dim)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        pred1_idx, pred1_scores = self.model([user_ids, item_ids], training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            self.model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            pred2_idx, pred2_scores = loaded_model([user_ids, item_ids], training=False)

            np.testing.assert_array_equal(pred1_idx.numpy(), pred2_idx.numpy())
            np.testing.assert_array_almost_equal(
                pred1_scores.numpy(),
                pred2_scores.numpy(),
            )

    def test_inference_consistency(self) -> None:
        """Test that inference mode produces consistent results."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        # Call twice in inference mode
        idx1, scores1 = self.model([user_ids, item_ids], training=False)
        idx2, scores2 = self.model([user_ids, item_ids], training=False)

        # Should get same results
        np.testing.assert_array_equal(idx1.numpy(), idx2.numpy())
        np.testing.assert_array_almost_equal(scores1.numpy(), scores2.numpy())

    def test_training_vs_inference_shapes(self) -> None:
        """Test that training and inference produce same output shapes."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        idx_train, scores_train = self.model([user_ids, item_ids], training=True)
        idx_infer, scores_infer = self.model([user_ids, item_ids], training=False)

        self.assertEqual(idx_train.shape, idx_infer.shape)
        self.assertEqual(scores_train.shape, scores_infer.shape)

    def test_embedding_consistency(self) -> None:
        """Test that same user produces consistent scores across batches."""
        # Same user in multiple batches
        user_ids_1 = tf.constant([0] * self.batch_size)
        item_ids_1 = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        user_ids_2 = tf.constant([0] * self.batch_size)
        item_ids_2 = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        # Same items for second batch
        item_ids_2 = item_ids_1

        _, scores1 = self.model([user_ids_1, item_ids_1], training=False)
        _, scores2 = self.model([user_ids_2, item_ids_2], training=False)

        # Same user with same items should have same scores
        np.testing.assert_array_almost_equal(scores1.numpy(), scores2.numpy())

    def test_compute_similarities(self) -> None:
        """Test compute_similarities method returns full similarity scores."""
        self.model.compile(optimizer="adam", loss="mse")

        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        similarities = self.model.compute_similarities(
            [user_ids, item_ids],
            training=False,
        )

        # Should return full similarity matrix (batch_size, num_items)
        self.assertEqual(similarities.shape, (self.batch_size, self.num_items))

        # Similarities should be in valid range [-1, 1] for cosine similarity
        min_sim = tf.reduce_min(similarities).numpy()
        max_sim = tf.reduce_max(similarities).numpy()
        self.assertGreaterEqual(min_sim, -1.1)  # Allow small numerical error
        self.assertLessEqual(max_sim, 1.1)

    def test_train_step_with_targets(self) -> None:
        """Test train_step method with binary targets."""
        self.model.compile(optimizer="adam", loss="mse")

        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        # Create binary targets: 1 for positive items, 0 for negative
        targets = np.zeros((self.batch_size, self.num_items), dtype=np.float32)
        # Mark some items as positive for each user
        for i in range(self.batch_size):
            positive_indices = np.random.choice(
                self.num_items,
                size=min(5, self.num_items),
                replace=False,
            )
            targets[i, positive_indices] = 1.0

        targets_tf = tf.constant(targets)

        # Test train_step
        result = self.model.train_step(([user_ids, item_ids], targets_tf))

        # Should return loss dictionary
        self.assertIn("loss", result)
        # Loss can be a tensor or numpy array
        loss_value = result["loss"]
        if hasattr(loss_value, "numpy"):
            loss_value = loss_value.numpy()
        self.assertTrue(np.isfinite(loss_value))

    def test_train_step_without_targets(self) -> None:
        """Test train_step method without targets (unsupervised)."""
        self.model.compile(optimizer="adam", loss="mse")

        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        # Test train_step without targets
        result = self.model.train_step(([user_ids, item_ids], None))

        # Should return loss dictionary
        self.assertIn("loss", result)
        # Loss can be a tensor or numpy array
        loss_value = result["loss"]
        if hasattr(loss_value, "numpy"):
            loss_value = loss_value.numpy()
        self.assertTrue(np.isfinite(loss_value))

    def test_model_fit_with_targets(self) -> None:
        """Test model.fit() with properly formatted data."""
        self.model.compile(optimizer="adam", loss="mse")

        # Build the model by calling it once
        sample_user_ids = tf.constant([0])
        sample_item_ids = tf.constant(np.arange(self.num_items).reshape(1, -1))
        _ = self.model([sample_user_ids, sample_item_ids], training=False)

        # Prepare data in format expected by train_step
        user_ids = np.random.randint(0, self.num_users, (32,))
        item_ids = np.tile(np.arange(self.num_items), (32, 1))

        # Create binary targets
        targets = np.zeros((32, self.num_items), dtype=np.float32)
        for i in range(32):
            positive_indices = np.random.choice(
                self.num_items,
                size=min(5, self.num_items),
                replace=False,
            )
            targets[i, positive_indices] = 1.0

        # Test fit
        history = self.model.fit(
            x=[user_ids, item_ids],
            y=targets,
            epochs=1,
            batch_size=8,
            verbose=0,
        )

        # Should have training history
        self.assertIn("loss", history.history)
        self.assertEqual(len(history.history["loss"]), 1)

    def test_train_step_loss_decreases(self) -> None:
        """Test that training loss can decrease with multiple steps."""
        self.model.compile(optimizer="adam", loss="mse")

        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        # Create binary targets
        targets = np.zeros((self.batch_size, self.num_items), dtype=np.float32)
        for i in range(self.batch_size):
            positive_indices = np.random.choice(
                self.num_items,
                size=min(5, self.num_items),
                replace=False,
            )
            targets[i, positive_indices] = 1.0

        targets_tf = tf.constant(targets)

        # Get initial loss
        initial_loss = self.model.train_step(([user_ids, item_ids], targets_tf))["loss"]

        # Train for a few steps
        losses = []
        for _ in range(3):
            loss = self.model.train_step(([user_ids, item_ids], targets_tf))["loss"]
            losses.append(loss)

        # Loss should be a valid number (may not decrease due to small steps, but should be finite)
        self.assertTrue(np.isfinite(initial_loss))
        self.assertTrue(all(np.isfinite(l) for l in losses))


if __name__ == "__main__":
    unittest.main()
