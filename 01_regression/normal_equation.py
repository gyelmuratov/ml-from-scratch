import numpy as np
# Normal Equation (ma'ruza, 22-slayd)
# θ = (XᵀX)⁻¹ Xᵀ y      — iteratsiyasiz, bitta formula bilan yechim
# θ = [b, w₁, w₂, ...]  — barcha parametrlar bitta vektorda

class NormalEquation:
    def __init__(self):
        self.w=None
        self.b=None

    def fit(self, X,y):
        m=X.shape[0]

        X_b = np.c_[np.ones(m), X]          # 1-qadam: 1 ustun qoshish
        XtX = X_b.T @ X_b                   # 2-qadam: XᵀX
        XtX_inv = np.linalg.pinv(XtX)        # 3-qadam: (XᵀX)⁻¹  — pinv: barqaror teskari matritsa
        Xty = X_b.T @ y                     # 4.qadam:  Xᵀy
        theta = XtX_inv @ Xty               # 5.qadam: θ = (XᵀX)⁻¹ · Xᵀy


        self.b = theta[0]
        self.w = theta[1:]
        return self

    def predict(self,X):
        return np.dot(X, self.w) + self.b       # ŷ = X·w + b


if __name__ == "__main__":
    # Qo'lda hisoblagan misol: x = [1, 2, 3], y = [2, 4, 5]
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 5])
    ne = NormalEquation().fit(X, y)
    print(f"b = {ne.b:.3f}, w = {ne.w[0]:.3f}")    