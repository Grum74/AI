import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = np.linspace(-3, 3, m)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, m)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X = X.reshape(-1, 1)
X_poly = poly_features.fit_transform(X, y)
Xtrain, Xtest, ytrain, ytest = train_test_split(X_poly, y, test_size=0.5, random_state=0)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)
fig, ax = plt.subplots()
ypred = lin_reg.predict(Xtest)
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
