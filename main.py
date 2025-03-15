import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats

# Load dataset
originalData = pd.read_csv('D600 Task 1 Dataset 1 Housing Information.csv')
df = pd.DataFrame(originalData)

print("PRINT INFO:")
print(df.info())
print("PRINT DESCRIPTIVE STATISTICS:")
print(df.describe().transpose().to_string())

# Selected columns
columns = ["Price", "SquareFootage", "NumBathrooms", "NumBedrooms", "BackyardSpace",
           "CrimeRate", "SchoolRating", "AgeOfHome", "DistanceToCityCenter", "EmploymentRate",
           "PropertyTaxRate", "RenovationQuality", "LocalAmenities", "TransportAccess",
           "PreviousSalePrice"]

# Drop missing values
df = df[columns].dropna()

# Identify Dependent and Independent Variables
X = df.drop(columns=["Price"])  # Independent variables
y = df["Price"]  # Dependent variable
# ------------------------------------- ASPECT C3: DATA VISUALIZATION ------------------------------------

# Univariate distribution
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

# Bivariate Analysis - Scatter plots of Independent Variables vs. Price
fig, axes = plt.subplots(5, 3, figsize=(15, 15))
axes = axes.flatten()

for i, var in enumerate(columns[1:]):  # Exclude 'Price' from scatter plots
    sns.scatterplot(x=df[var], y=df["Price"], ax=axes[i])
    axes[i].set_title(f"Price vs {var}")

plt.tight_layout()
plt.show()
# ---------------------------------- ASPECT D1: SPLIT THE DATA AND MODEL BUILDING -----------------------------
# Split the data into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add constant for intercept
X_train_const = sm.add_constant(X_train)

# Fit initial model
initialModel = sm.OLS(y_train, X_train_const).fit()
print("\nSUMMARY OF THE MODEL:")
print(initialModel.summary())
# -------------------------------- ASPECT D2: OPTIMIZATION - BACKWARD STEPWISE ELIMINATION ---------------------
# Backward Stepwise Elimination
def backward_stepwise(X_train, y_train, significance_level=0.05):
    while True:
        X_train_const = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train_const).fit()
        p_values = model.pvalues[1:]  # Exclude intercept
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            max_p_var = p_values.idxmax()
            X_train = X_train.drop(columns=[max_p_var])
        else:
            break
    return sm.OLS(y_train, sm.add_constant(X_train)).fit()

optimizedModel = backward_stepwise(X_train, y_train)
print("\nOPTIMIZED MODEL SUMMARY:")
print(optimizedModel.summary())
# ---------------------------------- ASPECT D3 and D4: MSE and SUMMARY -----------------------------
# Ensure X_train and X_test have the same selected features
selected_features = optimizedModel.model.exog_names  # Includes 'const'
selected_features.remove("const")  # Remove intercept

X_train_optimized = sm.add_constant(X_train[selected_features])
X_test_optimized = sm.add_constant(X_test[selected_features])

# Fit final model
final_model = LinearRegression()
final_model.fit(X_train_optimized.drop(columns=["const"]), y_train)

# Predictions
y_train_pred = final_model.predict(X_train_optimized.drop(columns=["const"]))
y_test_pred = final_model.predict(X_test_optimized.drop(columns=["const"]))

# Calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"\nOptimized Training MSE: {mse_train}")
print(f"Optimized Test MSE: {mse_test}")

# Final Regression Equation
intercept = final_model.intercept_
coefficients = final_model.coef_
equation = f"Price = {intercept:.2f}"
for feature, coef in zip(selected_features, coefficients):
    equation += f" + ({coef:.2f} * {feature})"

print("\nFinal Regression Equation:\n", equation)

# Model Performance
print("\nModel Metrics:")
print(f"R-squared: {optimizedModel.rsquared:.4f}")
print(f"Adjusted R-squared: {optimizedModel.rsquared_adj:.4f}")
print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# --------------------------------------- ASSUMPTIONS ---------------------------
# Linearity
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train_pred, y=y_train - y_train_pred)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values (Linearity Check)")
plt.show()

# Multicollinearity
correlation_matrix = X_train_optimized.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Homoscedasticity
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train_pred, y=y_train - y_train_pred)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values (Homoscedasticity Check)")
plt.show()

# Normality of Residuals
sns.histplot(y_train - y_train_pred, kde=True)
plt.title("Distribution of Residuals (Normality Check)")
plt.show()

# Q-Q Plot
sm.qqplot(y_train - y_train_pred, line='45', fit=True)
plt.title("Q-Q Plot for Normality Check")
plt.show()

# Shapiro-Wilk test for normality
residuals = y_train - final_model.predict(X_train_optimized.drop(columns=["const"]))
shapiro_test = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")

# Durbin-Watson test for autocorrelation
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(y_train - y_train_pred)
print(f"Durbin-Watson Statistic: {dw_stat:.4f}")
