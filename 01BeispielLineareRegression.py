import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# X (Feature) muss für scikit-learn ein 2D-Array sein
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# y (Target)
y = np.array([1, 4, 5.2, 7.1, 8.9])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

print("Ergebnisse")
print(f"Steigung (beta_1)         : {model.coef_[0]:.3f}")
print(f"Y-Achsenabschnitt (beta_0): {model.intercept_:.3f}")
print(f"Bestimmtheitsmaß (R^2)    : {r2:.3f}")

plt.style.use('ggplot')

plt.figure(figsize=(8, 5))

plt.scatter(X, y, color='#2ca02c', s=100, label='Echte Datenpunkte (y)', zorder=5)

plt.plot(X, y_pred, color='#d62728', linewidth=3, label='Regressionsgerade ($\hat{y}$)')

for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_pred[i]], color='gray', linestyle='--', alpha=0.5)

plt.xlabel('Unabhängige Variable (X)', fontsize=12, fontweight='bold')
plt.ylabel('Abhängige Variable (Y)', fontsize=12, fontweight='bold')
plt.title('Modell der Linearen Einfachregression', fontsize=14, pad=15)
plt.legend(fontsize=11, loc='upper left')

plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

plt.show()