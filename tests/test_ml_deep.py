from __future__ import annotations

import numpy as np
import pytest

from tankx.ml.models.base import Predictor
from tankx.ml.models.deep import GRUPredictor, LSTMPredictor, TransformerPredictor
from tankx.ml.windows import make_sequences


@pytest.fixture
def tiny_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 200
    n_feats = 4
    X = rng.normal(0.0, 1.0, size=(n, n_feats)).astype(np.float64)
    y_return = 0.2 * X[:, 0] + rng.normal(0.0, 1.0, size=n)
    y_direction = (y_return > 0).astype(np.float64)
    return (X, y_direction, y_return)

def test_make_sequences_shapes() -> None:
    X = np.arange(20 * 3, dtype=float).reshape(20, 3)
    y = np.arange(20, dtype=float)
    X_seq, y_seq = make_sequences(X, y, seq_len=5)
    assert X_seq.shape == (16, 5, 3)
    assert y_seq.shape == (16,)
    np.testing.assert_array_equal(X_seq[0], X[:5])
    assert y_seq[0] == y[4]

def test_make_sequences_rejects_short_input() -> None:
    X = np.zeros((3, 2))
    y = np.zeros(3)
    with pytest.raises(ValueError, match='seq_len'):
        make_sequences(X, y, seq_len=5)

def test_make_sequences_rejects_bad_dims() -> None:
    with pytest.raises(ValueError):
        make_sequences(np.zeros(10), np.zeros(10), seq_len=2)

@pytest.mark.parametrize('target_kind', ['direction', 'return'])
def test_lstm_fits_and_predicts(tiny_xy: tuple[np.ndarray, np.ndarray, np.ndarray], target_kind: str) -> None:
    X, y_dir, y_ret = tiny_xy
    y = y_dir if target_kind == 'direction' else y_ret
    m = LSTMPredictor(seq_len=8, hidden_size=4, epochs=2, batch_size=32, target_kind=target_kind, seed=0)
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape
    if target_kind == 'direction':
        assert ((pred >= 0) & (pred <= 1)).all()
    assert m.history

def test_lstm_satisfies_protocol() -> None:
    assert isinstance(LSTMPredictor(epochs=1), Predictor)

def test_lstm_predict_before_fit_returns_neutral() -> None:
    X = np.zeros((5, 3))
    m = LSTMPredictor(seq_len=4, target_kind='direction', epochs=1)
    out = m.predict(X)
    np.testing.assert_array_equal(out, np.full(5, 0.5))

def test_lstm_loss_decreases(tiny_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, _, y_ret = tiny_xy
    m = LSTMPredictor(seq_len=8, hidden_size=8, epochs=5, batch_size=32, target_kind='return', seed=0)
    m.fit(X, y_ret)
    assert m.history[-1]['train_loss'] <= m.history[0]['train_loss'] * 1.5

def test_gru_fits_and_predicts(tiny_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = tiny_xy
    m = GRUPredictor(seq_len=8, hidden_size=4, epochs=2, batch_size=32, target_kind='direction', seed=0)
    m.fit(X, y_dir)
    pred = m.predict(X)
    assert pred.shape == y_dir.shape

def test_gru_satisfies_protocol() -> None:
    assert isinstance(GRUPredictor(epochs=1), Predictor)

def test_transformer_fits_and_predicts(tiny_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = tiny_xy
    m = TransformerPredictor(seq_len=8, d_model=8, nhead=2, num_layers=1, epochs=2, batch_size=32, target_kind='direction', seed=0)
    m.fit(X, y_dir)
    pred = m.predict(X)
    assert pred.shape == y_dir.shape

def test_transformer_satisfies_protocol() -> None:
    assert isinstance(TransformerPredictor(epochs=1), Predictor)

def test_lstm_same_seed_same_predictions(tiny_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = tiny_xy
    a = LSTMPredictor(seq_len=8, hidden_size=4, epochs=3, batch_size=32, seed=42)
    b = LSTMPredictor(seq_len=8, hidden_size=4, epochs=3, batch_size=32, seed=42)
    a.fit(X, y_dir)
    b.fit(X, y_dir)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-06)

def test_lstm_handles_too_little_data() -> None:
    X = np.zeros((5, 3))
    y = np.zeros(5)
    m = LSTMPredictor(seq_len=32, target_kind='direction', epochs=1)
    m.fit(X, y)
    out = m.predict(X)
    assert out.shape == y.shape
    np.testing.assert_array_equal(out, np.full(5, 0.5))
