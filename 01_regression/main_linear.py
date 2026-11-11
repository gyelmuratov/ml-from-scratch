import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression              # ozimiz yozgan model
from metrics import mse, r2                                 # 1-qadamdagi metrikalar

# 1. Data : tajriba -> maosh 

np.random.seed(42)
X = np.random.uniform(0, 10, (50,1))                        # 50 ta xodimning tajribasi (0-10 yil)
y = 0.9 * X[:, 0] + 3.3 + np.random.normal(0, 0.8, 50)      # maosh = 0.9 * tajriba + shovqin

# 2. Modelni oqitish
model = LinearRegression(lr=0.01, n_iters=2000).fit(X,y)

# 3. Bashorat va baholash
y_pred = model.predict(X)
print(f"w = {model.w[0]:.3f}, b = {model.b:.3f}")          # kutilgan: w ≈ 0.9, b ≈ 3.3
print(f"MSE = {mse(y, y_pred):.3f}, R2 = {r2(y, y_pred):.3f}")
print(f"10 yillik tajriba → {model.predict(np.array([[10]]))[0]:.2f} mln so'm")

# 4. Grafiklar
plt.scatter(X, y, label="Ma'lumot")
plt.plot(X, y_pred, color="red", label="Model")
plt.xlabel("Tajriba (yil)")
plt.ylabel("Maosh (mln so'm)")
plt.legend()
plt.savefig("linear_fit.png")

plt.figure()
plt.plot(model.losses)
plt.title("Loss")
plt.xlabel("Iteratsiya")
plt.savefig("linear_loss.png")
print("Grafiklar saqlandi: linear_fit.png, linear_loss.png")