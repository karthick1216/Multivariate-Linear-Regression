# Implementation of Multivariate Linear Regression
## Developed by:KARTHICK S
## Register no:212224230114
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step 1: Prepare Data
- Load dataset, extract features (X) and target values (y).
### Step 2: Normalize Features
- Scale features using standardization or mean normalization for faster convergence.
### Step 3: Compute Cost Function
- Define ( J(\theta) ) using Mean Squared Error (MSE) to measure prediction error.
### Step 4: Apply Gradient Descent
- Update parameters ( \theta ) iteratively using learning rate and gradients.
### Step 5: Make Predictions
- Compute ( \hat{y} ) using learned parameters and new input features.

## Program:
```python
import pandas as pd
from sklearn import linear_model
df = pd.read_csv("/content/carsemission.csv")
X = df[['Weight', 'Volume']]
y = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(X, y)
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
input_data = pd.DataFrame({'Weight': [3300], 'Volume': [1300]})
predictedCO2 = regr.predict(input_data)
print('Predicted CO2 for the corresponding weight and volume:',predictedCO2)
```
## Output:
![image](https://github.com/user-attachments/assets/8707db0f-3a66-47ee-8418-46676d76820e)

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
