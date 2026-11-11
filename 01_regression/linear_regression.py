import numpy as np

# ŷ  = X·w + b                       bashorat
# J  = (1/2m) · Σ(ŷ − y)²            loss
# dw = (1/m) · Xᵀ·(ŷ − y)            w ni qaysi tomonga surish kerak
# db = (1/m) · Σ(ŷ − y)              b ni qaysi tomonga surish kerak
# w  = w − lr·dw                     yangilash
# b  = b − lr·db

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr=lr              # α - organish tezligi (learning rate): qadam hajmi
        self.n_iters=n_iters    # iteratsiyalar soni necha marta takrorlanadi
        self.w=None             # weights ogirliklar chiziq qiyaligi
        self.b=None             # b bias siljish chiziqning y oqini kesish nuqtasi 
        self.losses=[]          # har bir iteratsiyadagi J qiymati (grafik uchun)

    def fit(self, X,y):
        m,n=X.shape             # m namunalar soni, n feature lar soni
        self.w=np.zeros(n)      # boshlangich holat uchun: w=0
        self.b=0.0              # boshlangich holat uchun: b=0

        for _ in range(self.n_iters):
            # 1. Bashorat
            y_pred=np.dot(X, self.w) + self.b       # ŷ = X·w + b

            # 2.Xato va loss
            error=y_pred-y                           # (ŷ − y) — har bir nuqtadagi xato
            loss=np.sum(error**2) / (2*m)             # J = (1/2m) · Σ(ŷ − y)²
            self.losses.append(loss)                  # loss tarixini saqlaymiz

            # 3. Gradientlar
            dw = np.dot(X.T, error) / m             # dw = (1/m) · Xᵀ·(ŷ − y)
            db = np.sum(error) / m                  # db = (1/m) · Σ(ŷ − y)

            # 4. Yangilash (gradientga teskari tomonga qadam)
            self.w = self.w - self.lr * dw          # w = w − lr·dw
            self.b = self.b - self.lr * db          # b = b − lr·db

        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b           # ŷ = X·w + b (oqitilgan w, b bilan)