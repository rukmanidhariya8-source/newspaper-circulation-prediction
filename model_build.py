# model_build.py

# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Load the dataset
data = pd.read_csv('NewspaperData.csv')

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# Step 3: Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 4: Define input (X) and output (y) variables
X = data[['daily']]     # Independent variable
y = data['sunday']      # Dependent variable

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the model
model = LinearRegression()

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Predict on test data
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 10: Visualize results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Actual vs Predicted Sales based on Newspaper')
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 11: Save the model (optional)
import joblib
joblib.dump(model, 'newspaper_model.pkl')
print("\nModel saved as newspaper_model.pkl")
