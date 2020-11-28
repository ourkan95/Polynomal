import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x.values,y.values)

#plt.scatter(x,y)
#plt.plot(x,lr.predict(x))

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree = 2)

x_polynomial = pf.fit_transform(X)

lr2 = LinearRegression()

lr2.fit(x_polynomial,y)

#plt.scatter(X,Y)
#plt.plot(X,lr2.predict(pf.fit_transform(X)))

#svr (Kullanırkan scaler kullanmak zorunlu)

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")

svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli,y_olcekli)

plt.plot(x_olcekli,svr_reg.predict(x_olcekli))

print(svr_reg.predict([[6]]))
#print(lr2.predict([[6]]))






