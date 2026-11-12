import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from normal_equation import NormalEquation
from polynomial_features import polynomial_features, standardize
from metrics import mse

# 1. Data: parabola + shovqin 
np.random.seed(42)
x = np.random.uniform(0, 10, 30)                                # 30 ta nuqta (0-10) oraligida 
y = 0.3 * x**2 - 2*x +5 + np.random.normal(0, 1.5, 30)          # haqiqiy bogliqlik - parabola

# 2. Train / test ga bolish (70 % / 30 % )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 3. Grafik uchun silliq x qiymatlari 
x_plot = np.linspace(0, 10, 300)
plt.scatter(x_train, y_train, label="Train")
plt.scatter(x_test, y_test, label="Test", marker="x")

# 4. Asosiy tajriba: d = 1, 2, 15 (ma'ruza, 28-slayd)
print(f"{'Daraja':>6} | {'Train MSE':>10} | {'Test MSE':>10}")
print("-" * 32)

for d in [1,2,15]:
    # Featurelar: x → [x, x², ..., x^d]
    X_train = polynomial_features(x_train, d)
    X_test = polynomial_features(x_test,d)

    # Scalling: mean va std faqat train dan oladi
    X_train, mean, std = standardize(X_train)
    X_test, _, _ = standardize(X_test, mean, std)

    # O'qitish va baholash
    model = NormalEquation().fit(X_train, y_train)
    train_mse = mse(y_train, model.predict(X_train))
    test_mse = mse(y_test, model.predict(X_test))
    print(f"{d:>6} | {train_mse:>10.3f} | {test_mse:>10.3f}")

    # Model egri chizig'ini chizish
    X_plot, _, _ = standardize(polynomial_features(x_plot, d), mean, std)
    plt.plot(x_plot, model.predict(X_plot), label=f"d = {d}")

plt.ylim(-5, 25)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polinomial regressiya: daraja tanlash")
plt.legend()
plt.savefig("polynomial_degrees.png")
print("\nGrafik saqlandi: polynomial_degrees.png")
