
"""
5-qadam: mini-loyiha (ma’ruzaning 33-slaydi). Avval masalani, keyin kodni ko’ramiz.

1. Masala nima?

Bu bosqichda yangi narsa o’rganmaymiz. 1–4-qadamlarda yozgan hamma narsani bitta real masalada birlashtiramiz.

Masala: uy maydoniga qarab uy narxini bashorat qilish.

Uy maydoni	Narx
40 m²	~300 mln so’m
80 m²	~400 mln so’m
120 m²	~650 mln so’m
150 m²	~900 mln so’m

Qiziq jihati: narx maydonga tekis emas, tezlashib oshadi. Kichik uylarda har bir qo’shimcha m² narxni kam oshiradi, katta uylarda esa ko’p. Ya’ni bog’liqlik to’g’ri chiziq emas, egri.

Savol: bu masala uchun qaysi model yaxshiroq, chiziqli yoki polinomial? Javobni taxmin bilan emas, metrikalar bilan isbotlaymiz.

2. Nima qilamiz
#	Qadam	Qayerdan olingan
1	100 ta uy haqida data yaratamiz	
2	Train (80%) va test (20%) ga bo’lamiz	7-slayd
3	Chiziqli modelni o’qitamiz (d=1)	3-qadam: Normal Equation
4	Polinomial modelni o’qitamiz (d=2)	4-qadam
5	Ikkalasini test da MSE, RMSE, R² bilan baholaymiz	1-qadam: metrics.py
6	Natijalarni sklearn bilan tekshiramiz	21, 27, 32-slaydlar
7	Grafik chizamiz va xulosa qilamiz	33-slayd
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SkLinear
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from normal_equation import NormalEquation
from polynomial_features import polynomial_features, standardize
from metrics import mse, rmse, r2

# 1. Data: uy maydoni -> narx
np.random.seed(42)
area = np.random.uniform(30, 150, 100)                                  # 100 ta uy, 30–150 m²
price = 0.04 * area**2 - 2 * area + 300 + np.random.normal(0,40,100)    # narx — egri bog'liqlik + shovqin

# 2. Train / test (80 % / 20 %)
a_train, a_test, p_train, p_test = train_test_split(area, price, test_size=0.2, random_state=42)

# Ikkala modelni oqitish va baholash
models = {}
print(f"{'Model':<18} | {'MSE':>9} | {'RMSE':>7} | {'R2':>6}")
print("-" * 50)

for name, d in [("Chiziqli", 1), ("Polinomial (d=2)", 2)]:
    # Featurelar va scalling (mean, std, faqat train dan)
    X_train , mean, std = standardize(polynomial_features(a_train, d))
    X_test, _, _ = standardize(polynomial_features(a_test, d), mean, std)

    # Oqitish (Normal Equation)
    model = NormalEquation().fit(X_train, p_train)

    # Test da baholash
    pred = model.predict(X_test)
    print(f"{name:<18} | {mse(p_test, pred):>9.1f} | {rmse(p_test, pred):>7.1f} | {r2(p_test, pred):>6.3f}")

    models[name] = (model, mean, std, d)

# 6. sklearn bilan tekshirish

sk_linear = SkLinear().fit(a_train.reshape(-1,1), p_train)
sk_poly = make_pipeline(PolynomialFeatures(2), SkLinear()).fit(a_train.reshape(-1, 1), p_train)

print("\ nsklearn bilan tekshirish:")
for name, sk_model in [("Chiziqli", sk_linear), ("Polinomial (d=2)", sk_poly)]:
    pred = sk_model.predict(a_test.reshape(-1,1))
    print(f"{name:<18} | MSE = {mean_squared_error(p_test, pred):.1f} | R2 = {r2_score(p_test, pred):.3f}")

# Yangi uy uchun bashorat
model, mean, std, d = models["Polinomial (d=2)"]
X_new, _, _ = standardize(polynomial_features([100], d), mean, std)
print(f"\n100 m² uy narxi (polinomial model): {model.predict(X_new)[0]:.0f} mln so'm")

# ===== 7. Grafik =====
x_plot = np.linspace(30, 150, 300)
plt.scatter(a_train, p_train, label="Train", alpha=0.6)
plt.scatter(a_test, p_test, label="Test", marker="x")
for name, (model, mean, std, d) in models.items():
    X_plot, _, _ = standardize(polynomial_features(x_plot, d), mean, std)
    plt.plot(x_plot, model.predict(X_plot), label=name, linewidth=2)
plt.xlabel("Uy maydoni (m²)")
plt.ylabel("Narx (mln so'm)")
plt.title("Uy narxi: chiziqli va polinomial model")
plt.legend()
plt.savefig("house_price.png")
print("Grafik saqlandi: house_price.png")
