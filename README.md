# EXP3-Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries in python required for finding Gradient Design.
2. Read the dataset file and check any null value using .isnull() method.
3. Declare the default variables with respective values for linear regression.
4. Calculate the loss using Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.
7. Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.
## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: MUSFIRA MAHJABEEN M
RegisterNumber:  212223230130
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None) 
print(data.head())

X = (data.iloc[1:, :-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()

y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
#### X & Y VALUES:
![Screenshot 2024-03-31 144049](https://github.com/MOHAMEDAHSAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139331378/281aa602-1cc6-4016-8fb2-fc553fcd9157)
![Screenshot 2024-03-31 144057](https://github.com/MOHAMEDAHSAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139331378/d0446f88-bd76-43b7-9cfd-37d0583883e0)


#### X SCALED & Y SCALED VALUES:
![Screenshot 2024-03-31 144115](https://github.com/MOHAMEDAHSAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139331378/7ce18852-131e-4da0-94e5-f898d9039297)
![Screenshot 2024-03-31 144106](https://github.com/MOHAMEDAHSAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139331378/e54d9e71-9c57-48d8-bbba-0a6042ac8855)


#### PREDICTED VALUE:
![Screenshot 2024-03-31 144128](https://github.com/MOHAMEDAHSAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139331378/f249b2ea-7c83-4b23-995d-9f3f7d1d4af9)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
