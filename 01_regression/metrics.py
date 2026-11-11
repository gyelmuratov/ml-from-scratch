import numpy as np


def mse(y_true, y_pred):
    """O'rtacha kvadratik xato: (1/m) · Σ(yᵢ − ŷᵢ)²"""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """MSE ning ildizi — asl birlikdagi xato"""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """O'rtacha absolyut xato: (1/m) · Σ|yᵢ − ŷᵢ|"""
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    """Determinatsiya koeffitsiyenti: 1 − SSE / SST"""
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - sse / sst


if __name__ == "__main__":
    y = np.array([10, 15, 20, 25, 30])
    y_pred = np.array([12, 14, 17, 26, 29])

    print("MSE: ", mse(y, y_pred))    # 3.2
    print("RMSE:", rmse(y, y_pred))   # ≈ 1.789
    print("MAE: ", mae(y, y_pred))    # 1.6
    print("R2:  ", r2(y, y_pred))     # 0.936

    # Slayddagi mashq: qaysi model yaxshiroq?
    y_pred2 = np.array([11, 16, 19, 24, 31])
    print("\n2-model MSE:", mse(y, y_pred2))
    print("2-model R2: ", r2(y, y_pred2))