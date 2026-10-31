import numpy as np

from models.output_strength_calibrator import apply_receiver_calibrator, receiver_feature_matrix


class ConstantRegressor:
    def predict(self, features):
        return np.full(len(features), 0.8, dtype=np.float32)


def test_receiver_calibration_preserves_late_time_predictions():
    gates = 869
    rng = np.random.default_rng(7)
    noisy = rng.normal(-6.0, 0.3, (2, 11, gates)).astype(np.float32)
    pred = (noisy + rng.normal(0.0, 0.1, noisy.shape)).astype(np.float32)

    features = receiver_feature_matrix(noisy, pred)
    corrected, alpha = apply_receiver_calibrator(
        noisy, pred, ConstantRegressor(), early_only=True, savgol_window=101
    )

    assert features.shape == (22, 34)
    np.testing.assert_array_equal(alpha, np.full((2, 11), 0.8, dtype=np.float32))
    assert np.any(corrected[:, :, :int(0.65 * gates)] != pred[:, :, :int(0.65 * gates)])
    np.testing.assert_array_equal(corrected[:, :, int(0.70 * gates):], pred[:, :, int(0.70 * gates):])
