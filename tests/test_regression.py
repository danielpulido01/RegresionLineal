"""Verifica directamente las funciones implementadas en el notebook."""

import json
from pathlib import Path
import unittest

import numpy as np


notebook = json.loads(
    (Path(__file__).resolve().parents[1] / "main.ipynb").read_text(encoding="utf-8")
)
namespace = {"np": np}
source = next(cell["source"] for cell in notebook["cells"]
              if cell.get("id") == "regression-functions")
exec(compile("".join(source), "main.ipynb:regression-functions", "exec"), namespace)
costo = namespace["costo"]
mse_manual = namespace["mse_manual"]
gradiente = namespace["gradiente"]
batch = namespace["batch_gradient_descent"]
mini_batch = namespace["mini_batch_gradient_descent"]


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack([np.ones(5), [-2., -1., 0., 1., 2.]])
        self.y = self.X @ np.array([3., 2.])

    def test_gradient_matches_finite_differences(self):
        beta = np.array([0.7, -0.4])
        epsilon = 1e-6
        numerical = []
        for direction in np.eye(2) * epsilon:
            numerical.append((costo(self.X, self.y, beta + direction)
                              - costo(self.X, self.y, beta - direction)) / (2 * epsilon))
        np.testing.assert_allclose(gradiente(self.X, self.y, beta), numerical, rtol=1e-7)

    def test_manual_mse_matches_hand_calculation(self):
        # Errores: 1, -2, 3; suma de cuadrados: 14; MSE: 14/3.
        self.assertAlmostEqual(mse_manual(np.array([1., 2., 3.]),
                                         np.array([2., 0., 6.])), 14 / 3)

    def test_mse_is_recorded_after_every_weight_update(self):
        X, y = np.ones((3, 1)), np.full(3, 2.)
        X_val, y_val = np.ones((2, 1)), np.full(2, 3.)
        # Batch: beta = 0, 0.2. Mini Batch: beta = 0, 0.2, 0.38.
        for trainer, betas in [(batch, [0., 0.2]), (mini_batch, [0., 0.2, 0.38])]:
            options = {"batch_size": 2} if trainer is mini_batch else {}
            result = trainer(X, y, X_val, y_val, alpha=0.1, epocas=1, **options)
            history = result["historial_mse"]
            np.testing.assert_allclose(history["train"], (2 - np.array(betas)) ** 2)
            np.testing.assert_allclose(history["val"], (3 - np.array(betas)) ** 2)
            self.assertEqual(len(history["train"]), result["actualizaciones"] + 1)

    def test_history_evaluates_full_training_set_after_each_mini_batch(self):
        seed, alpha, size = 7, 0.05, 2
        beta = np.zeros(2)
        expected = [mse_manual(self.y, self.X @ beta)]
        order = np.random.default_rng(seed).permutation(len(self.y))
        for start in range(0, len(self.y), size):
            indices = order[start:start + size]
            beta -= alpha * gradiente(self.X[indices], self.y[indices], beta)
            expected.append(mse_manual(self.y, self.X @ beta))
        result = mini_batch(self.X, self.y, self.X, self.y, alpha=alpha,
                            epocas=1, batch_size=size, semilla=seed)
        np.testing.assert_allclose(result["historial_mse"]["train"], expected)

    def test_batch_converges_to_known_solution(self):
        result = batch(self.X, self.y, self.X, self.y, alpha=0.1, epocas=200)
        np.testing.assert_allclose(result["beta"], [3., 2.], atol=1e-7)
        self.assertTrue(np.all(np.diff(result["historial"]["train"]) <= 0))
        self.assertEqual(len(result["historial"]["val"]), 201)
        self.assertEqual(result["actualizaciones"], 200)

    def test_full_size_mini_batch_matches_batch(self):
        a = batch(self.X, self.y, self.X, self.y, alpha=0.1, epocas=30)
        b = mini_batch(self.X, self.y, self.X, self.y, alpha=0.1,
                       epocas=30, batch_size=5)
        np.testing.assert_allclose(a["beta"], b["beta"], atol=1e-12)
        self.assertEqual(b["actualizaciones"], 30)

    def test_last_incomplete_batch_uses_its_actual_size(self):
        # Tres objetivos constantes y lotes de 2: el último contiene un solo dato.
        X, y = np.ones((3, 1)), np.full(3, 2.)
        result = mini_batch(X, y, X, y, alpha=0.1, epocas=1, batch_size=2)
        # Primera actualización: 0 -> 0.2; segunda: 0.2 -> 0.38.
        np.testing.assert_allclose(result["beta"], [0.38])
        self.assertEqual(result["actualizaciones"], 2)

    def test_mini_batch_converges_and_is_reproducible(self):
        args = (self.X, self.y, self.X, self.y)
        options = dict(alpha=0.05, epocas=200, batch_size=2, semilla=19)
        first = mini_batch(*args, **options)
        second = mini_batch(*args, **options)
        np.testing.assert_allclose(first["beta"], [3., 2.], atol=1e-7)
        np.testing.assert_array_equal(first["beta"], second["beta"])
        self.assertEqual(first["historial"], second["historial"])

    def test_validation_does_not_change_parameters(self):
        for trainer in (batch, mini_batch):
            a = trainer(self.X, self.y, self.X, self.y, alpha=0.1, epocas=10)
            b = trainer(self.X, self.y, self.X, self.y + 100, alpha=0.1, epocas=10)
            np.testing.assert_array_equal(a["beta"], b["beta"])


if __name__ == "__main__":
    unittest.main()
