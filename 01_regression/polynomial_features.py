import numpy as np

def polynomial_features(x, degree):
    """
    x-> [x, x2, x3, ... , x degree]
    """
    x = np.asarray(x).ravel()                                       # x ni bir olchamli masivga keltiramiz
    return np.column_stack([x ** d for d in range(1, degree + 1)])  # har bir daraja alohida ustun

def standardize(X, mean=None, std=None):
    """
    Har bir ustunni (X-mean) / std korinishiga keltiradi
    """
    if mean is None:                                # train uchun: mean va std hisoblaymiz
        mean = X.mean(axis=0)                       # har bir ustunning ortachasi
        std = X.std(axis=0)                         # har bir ustunning standart ogishi
    return (X-mean) / std, mean, std                # test uchun: train dagi mean, std beriladi