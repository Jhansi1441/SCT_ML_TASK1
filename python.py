import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Create the dataset with limited room numbers and square footage up to 700
np.random.seed(0)
data = {
    'Square_Footage': np.random.randint(100, 700, 50),  # Square footage limited to 700
    'Bedrooms': np.random.randint(1, 4, 50),            # Bedrooms limited to 3
    'Bathrooms': np.random.randint(1, 3, 50),           # Bathrooms limited to 2
    'Price': np.random.randint(50000, 500000, 50)       # Price range for smaller homes
}
df = pd.DataFrame(data)

# Step 2: Data exploration
print(df.head())

# Step 3: Visualize the data
sns.pairplot(df)
plt.show()

# Step 4: Prepare the data
X = df[['Square_Footage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Coefficients of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 9: Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
