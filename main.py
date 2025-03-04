import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

originalData=pd.read_csv('D600 Task 1 Dataset 1 Housing Information.csv')
df=pd.DataFrame(originalData)
print("PRINT INFO:")
print(df.info())
print("PRINT DESCRIPTIVE STATISTICS:")
print(df.describe().transpose().to_string())


# ------------------------------------- ASPECTS C1 and C2: DATA PREPARATION ------------------------------------
# Selected columns
columns = ["Price","SquareFootage", "NumBathrooms", "NumBedrooms", "BackyardSpace",
                 "CrimeRate", "SchoolRating","AgeOfHome","DistanceToCityCenter", "EmploymentRate",
                 "PropertyTaxRate", "RenovationQuality", "LocalAmenities", "TransportAccess",
                 "PreviousSalePrice"]
# drop missing values
df=df[columns].dropna()

# Identify Dependent and Independent Variables
X = df.drop(columns=["Price"])  # Independent variables
y = df["Price"]  # Dependent variable

print("\nDESCRIPTIVE STATISTICS AFTER REMOVING MISSING VALUES:")
print(df[columns].describe().transpose().to_string())
dfColumns=df[columns]


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

#Fit initial model
initialModel = sm.OLS(y_train, X_train_const).fit()
print("\nSUMMARY OF THE MODEL:")
print(initialModel.summary())
print("\n")
# -------------------------------- ASPECT D2: OPTIMIZATION - BACKWARD STEPWISE ELIMINATION ---------------------
X_train_optimized = X_train_const.copy() #copy of data with intercept

while True:
    optimizedModel=sm.OLS(y_train, X_train_optimized).fit()
    pValues=optimizedModel.pvalues
    pMax=pValues.max() #find the highest p-value
    if pMax > 0.05:  # Check if pMax exceeds threshold
        excludedVariable = optimizedModel.pvalues.idxmax()  # Find variable with max p-value
        print(f"REMOVING {excludedVariable} with p-value: {pMax:.4f}")
        X_train_optimized = X_train_optimized.drop(columns=[excludedVariable])  # Remove it
    else:
        break  # Stop when all p-values are below threshold
print("\n OPTIMIZED MODEL SUMMARY:")
print(optimizedModel.summary())

# ---------------------------------- ASPECT D3 and D4: MSE  -----------------------------
X_train_optimized = X_train_const.copy()
X_test_optimized = X_test[X_train_optimized.columns.drop("const")]

final_model = LinearRegression()
final_model.fit(X_train_optimized.drop(columns=["const"]), y_train)

y_train_pred = final_model.predict(X_train_optimized.drop(columns=["const"]))
y_test_pred = final_model.predict(X_test_optimized)

# Calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"\nOptimized Training MSE: {mse_train}")
print(f"Optimized Test MSE: {mse_test}")

# Save optimized datasets
y_train.to_csv("training.csv", index=False)
y_test.to_csv("test.csv", index=False)

#----------------------------------- SUMMARY -----------------------------------------

# Final Regression Equation
intercept = final_model.intercept_
coefficients = final_model.coef_
equation = f"Price = {intercept:.2f}"
for feature, coef in zip(X_train_optimized.columns.drop("const"), coefficients):
    equation += f" + ({coef:.2f} * {feature})"

print("\nFinal Regression Equation:\n", equation)

# Model Performance
print("\nModel Metrics:")
print(f"R-squared: {optimizedModel.rsquared:.4f}")
print(f"Adjusted R-squared: {optimizedModel.rsquared_adj:.4f}")
print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

