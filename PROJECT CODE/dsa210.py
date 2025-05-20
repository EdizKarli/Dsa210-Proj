import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from google.colab import drive

drive.mount('/content/drive')

df = pd.read_excel('/content/drive/MyDrive/dsa210data.xlsx')

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check column names
print("Available columns:", df.columns.tolist())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check first few rows
print("\nFirst few rows:")
print(df.head())

# HISTOGRAM VISUALIZATIONS
print("\n" + "="*50)
print("HISTOGRAM VISUALIZATIONS")
print("="*50)

# Histograms for numerical columns (excluding Homework/Project)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

custom_colors = {
    'HW/Proj Exists': 'green',
    "HW/Proj Doesn't Exist": 'red'
}

# Get a color palette for the rest of the distributions
palette = sns.color_palette("Set2", len(numerical_cols))  # or "tab10"

# Calculate number of rows needed (3 plots per row)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_cols = len(numerical_cols)
num_rows = (num_cols + 2) // 3  # Ceiling division
temp_df = pd.DataFrame({'HW/Proj Label': df['Homework/Project'].map({1: 'HW/Proj Exists', 0: "HW/Proj Doesn't Exist"})})
# Create histograms
plt.figure(figsize=(15, 5 * num_rows))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(num_rows, 3, i)

    if col in ['School Time', 'Travel Time', 'Sleep Time']:
        # Convert minutes to hours
        temp_data = pd.DataFrame({col: df[col] / 60})
        sns.histplot(data=temp_data, x=col, color=palette[i % len(palette)])
        plt.xlabel(f'{col} (Hours)')

    elif col == 'Homework/Project':
        # Map binary values to labels
        temp_data = df[col].map({1: 'HW/Proj Exists', 0: "HW/Proj Doesn't Exist"})
        sns.countplot(data=temp_df, x='HW/Proj Label', hue='HW/Proj Label', palette=custom_colors, legend=False)
        plt.xlabel('Homework/Project Status')

    else:
        sns.histplot(data=df, x=col, color=palette[i % len(palette)])

    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("BOXPLOT VISUALIZATIONS")
print("="*50)

# Box plots for numerical columns (excluding Homework/Project)
plt.figure(figsize=(15, 5 * num_rows))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(num_rows, 3, i)
    if col in ['School Time', 'Travel Time', 'Sleep Time']:
        # Convert to hours
        temp_data = pd.DataFrame({col: df[col] / 60})
        sns.boxplot(data=temp_data, y=col, color='purple')
        plt.ylabel(f'{col} (Hours)')
    else:
        sns.boxplot(data=df, y=col, color='purple')
    plt.title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()

# CORRELATION VISUALIZATIONS
print("\n" + "="*50)
print("CORRELATION VISUALIZATIONS")
print("="*50)

# Create correlation matrices for different combinations
# 1. Instagram correlations
instagram_corr = df[['School Time', 'Travel Time', 'Sleep Time', 'Sleep Quality',
                    'Social Interaction', 'Instagram Usage']].corr()

# 2. Reddit correlations
reddit_corr = df[['School Time', 'Travel Time', 'Sleep Time', 'Sleep Quality',
                  'Social Interaction', 'Reddit Usage']].corr()

# 3. Combined correlations
combined_corr = df[['Instagram Usage', 'Reddit Usage', 'School Time', 'Travel Time',
                   'Sleep Time', 'Sleep Quality', 'Social Interaction']].corr()

# Visualize correlation matrices in a 3x1 layout
plt.figure(figsize=(15, 15))

# Instagram correlations
plt.subplot(3, 1, 1)
sns.heatmap(instagram_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlations with Instagram Usage')

# Reddit correlations
plt.subplot(3, 1, 2)
sns.heatmap(reddit_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlations with Reddit Usage')

# Combined correlations
plt.subplot(3, 1, 3)
sns.heatmap(combined_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Combined Correlations')

plt.tight_layout()
plt.show()

# Calculate average metrics for different activities
daily_averages = df.agg({
    'School Time': lambda x: (x.mean()/60).round(2),  # Convert to hours
    'Travel Time': lambda x: (x.mean()/60).round(2),  # Convert to hours
    'Sleep Time': lambda x: (x.mean()/60).round(2),   # Convert to hours
    'Instagram Usage': 'mean',
    'Reddit Usage': 'mean'
}).round(2)

print("Daily Averages (Time in Hours):")
print(daily_averages)

fig, axes = plt.subplots(6, 2, figsize=(16, 30))  # 6 rows, 2 cols; bigger height
axes = axes.flatten()  # Flatten to make indexing easier

# Define your x and y variables with labels
plot_info = [
    ('School Time', 'Instagram Usage'),
    ('School Time', 'Reddit Usage'),
    ('Travel Time', 'Instagram Usage'),
    ('Travel Time', 'Reddit Usage'),
    ('Sleep Time', 'Instagram Usage'),
    ('Sleep Time', 'Reddit Usage'),
    ('Sleep Quality', 'Instagram Usage'),
    ('Sleep Quality', 'Reddit Usage'),
    ('Social Interaction', 'Instagram Usage'),
    ('Social Interaction', 'Reddit Usage'),
    ('Homework/Project', 'Instagram Usage'),
    ('Homework/Project', 'Reddit Usage'),
]

# Plot each pair
for i, (x_var, y_var) in enumerate(plot_info):
    # Convert minutes to hours for time-based columns
    if x_var in ['School Time', 'Travel Time', 'Sleep Time']:
        x_data = df[x_var] / 60
        x_label = f"{x_var} (Hours)"
    else:
        x_data = df[x_var]
        x_label = x_var

    axes[i].scatter(x_data, df[y_var])
    axes[i].set_xlabel(x_label)
    axes[i].set_ylabel(y_var)
    axes[i].set_title(f"{x_var} vs {y_var}")

plt.tight_layout()
plt.show()

# List of pairs to analyze
pairs = [
    ('School Time', 'Instagram Usage'),
    ('School Time', 'Reddit Usage'),
    ('Travel Time', 'Instagram Usage'),
    ('Travel Time', 'Reddit Usage'),
    ('Sleep Time', 'Instagram Usage'),
    ('Sleep Time', 'Reddit Usage'),
    ('Sleep Quality', 'Instagram Usage'),
    ('Sleep Quality', 'Reddit Usage')
]

# Perform correlation analysis and create scatter plots for each pair
for time_var, usage_var in pairs:
    # Calculate correlation
    corr_coef, p_value = stats.pearsonr(df[time_var], df[usage_var])

    # Create scatter plot with regression line
    plt.figure(figsize=(10, 6))
    if time_var in ['School Time', 'Travel Time', 'Sleep Time']:
        # Convert time to hours for display
        sns.regplot(x=df[time_var]/60, y=df[usage_var], data=df)
        plt.xlabel(f'{time_var} (Hours)')
    else:
        sns.regplot(x=df[time_var], y=df[usage_var], data=df)
        plt.xlabel(time_var)

    plt.ylabel(usage_var)
    plt.title(f'Correlation between {time_var} and {usage_var}\n'
              f'Correlation: {corr_coef:.3f}, P-value: {p_value:.3f}')
    plt.show()

    # Print detailed statistics
    print(f"\nCorrelation Analysis: {time_var} vs {usage_var}")
    print(f"Correlation coefficient: {corr_coef:.3f}")
    print(f"P-value: {p_value:.3f}")
    if p_value < 0.05:
        print("This correlation is statistically significant (p < 0.05)")
    else:
        print("This correlation is not statistically significant (p >= 0.05)")
    print("-" * 50)

#additional pairs

additional_pairs = [
    ('Social Interaction', 'Instagram Usage'),
    ('Social Interaction', 'Reddit Usage'),
    ('Homework/Project' ,'Instagram Usage'),
    ('Homework/Project', 'Reddit Usage'),
]

# Perform correlation analysis and create scatter plots for each new pair
for time_var, usage_var in additional_pairs:
    # Calculate correlation
    corr_coef, p_value = stats.pearsonr(df[time_var], df[usage_var])

    # Create scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df[time_var], y=df[usage_var], data=df)
    plt.xlabel(time_var)
    plt.ylabel(usage_var)
    plt.title(f'Correlation between {time_var} and {usage_var}\n'
              f'Correlation: {corr_coef:.3f}, P-value: {p_value:.3f}')
    plt.show()

    # Print detailed statistics
    print(f"\nCorrelation Analysis: {time_var} vs {usage_var}")
    print(f"Correlation coefficient: {corr_coef:.3f}")
    print(f"P-value: {p_value:.3f}")
    if p_value < 0.05:
        print("This correlation is statistically significant (p < 0.05)")
    else:
        print("This correlation is not statistically significant (p >= 0.05)")
    print("-" * 50)

# STATISTICAL TEST RESULTS
print("\n" + "="*50)
print("STATISTICAL TEST RESULTS")
print("="*50)

# Define variables to test against Instagram and Reddit
variables_to_test = ['School Time', 'Travel Time', 'Sleep Time', 'Sleep Quality', 'Social Interaction']

# Function to perform statistical tests
def perform_statistical_tests(time_var, usage_var):
    print(f"\n{'='*50}")
    print(f"Statistical Tests for {time_var} vs {usage_var}")
    print(f"{'='*50}")

    # 1. Chi-Square Test
    time_median = df[time_var].median()
    usage_median = df[usage_var].median()

    contingency_table = pd.crosstab(
        df[time_var] > time_median,
        df[usage_var] > usage_median
    )

    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
    print("\nChi-Square Test:")
    print(f"Chi2 statistic: {chi2:.3f}")
    print(f"P-value: {p_chi:.3f}")

    # 2. T-Test
    high_time = df[df[time_var] > time_median][usage_var]
    low_time = df[df[time_var] <= time_median][usage_var]

    t_stat, p_t = stats.ttest_ind(high_time, low_time)
    print("\nT-Test:")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_t:.3f}")

    # Print interpretation
    print("\nInterpretation:")
    if p_chi < 0.05:
        print("- Chi-Square: Significant relationship found")
    if p_t < 0.05:
        print("- T-Test: Significant difference in means found")
    if p_chi >= 0.05 and p_t >= 0.05:
        print("- No significant relationships found in any test")

# Perform tests for Instagram
print("\n" + "="*50)
print("STATISTICAL TESTS FOR INSTAGRAM USAGE")
print("="*50)
for var in variables_to_test:
    perform_statistical_tests(var, 'Instagram Usage')


# Perform tests for Reddit
print("\n" + "="*50)
print("STATISTICAL TESTS FOR REDDIT USAGE")
print("="*50)
for var in variables_to_test:
    perform_statistical_tests(var, 'Reddit Usage')

custom_palette = {"Doesn't Exist": 'red', 'Exists': 'green'}

# Ensure the label column exists
df['HW/Proj Label'] = df['Homework/Project'].map({0: "Doesn't Exist", 1: 'Exists'})

# Reddit Usage
sns.boxplot(data=df, x='HW/Proj Label', y='Reddit Usage', hue='HW/Proj Label', palette=custom_palette, legend=False)
plt.title('Reddit Usage vs Homework/Project Presence')
plt.ylabel('Reddit Usage (e.g., minutes)')
plt.xlabel('Homework/Project')
plt.show()

# Instagram Usage
sns.boxplot(data=df, x='HW/Proj Label', y='Instagram Usage', hue='HW/Proj Label', palette=custom_palette, legend=False)
plt.title('Instagram Usage vs Homework/Project Presence')
plt.ylabel('Instagram Usage (e.g., minutes)')
plt.xlabel('Homework/Project')
plt.show()

features = ['School Time', 'Travel Time', 'Sleep Time', 'Sleep Quality',
            'Social Interaction', 'Homework/Project']
X = df[features]
y = df['Youtube Usage']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression (YouTube Usage):")
print(f" - RMSE: {rmse_lr:.2f}")
print(f" - MAE: {mae_lr:.2f}")
print(f" - R² Score: {r2_lr:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='skyblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual YouTube Usage")
plt.ylabel("Predicted YouTube Usage")
plt.title("Linear Regression: Actual vs Predicted YouTube Usage")
plt.grid(True)
plt.show()

print("---------------------------------------------------------------------")
print()

model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Regression (YouTube Usage):")
print(f" - RMSE: {rmse_dt:.2f}")
print(f" - MAE: {mae_dt:.2f}")
print(f" - R² Score: {r2_dt:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_dt, alpha=0.7, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual YouTube Usage")
plt.ylabel("Predicted YouTube Usage")
plt.title("Decision Tree: Actual vs Predicted YouTube Usage")
plt.grid(True)
plt.show()

print("---------------------------------------------------------------------")
print()

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regression (YouTube Usage):")
print(f" - RMSE: {rmse_rf:.2f}")
print(f" - MAE: {mae_rf:.2f}")
print(f" - R² Score: {r2_rf:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual YouTube Usage")
plt.ylabel("Predicted YouTube Usage")
plt.title("Random Forest: Actual vs Predicted YouTube Usage")
plt.grid(True)
plt.show()

# 1. Predicting Instagram Usage
X_instagram = df[features]
y_instagram = df['Instagram Usage']
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_instagram, y_instagram, test_size=0.2, random_state=42)

lr_i = LinearRegression()
lr_i.fit(X_train_i, y_train_i)
y_pred_lr_i = lr_i.predict(X_test_i)

print("Instagram Usage - Linear Regression:")
print(f" - RMSE: {np.sqrt(mean_squared_error(y_test_i, y_pred_lr_i)):.2f}")
print(f" - MAE: {mean_absolute_error(y_test_i, y_pred_lr_i):.2f}")
print(f" - R² Score: {r2_score(y_test_i, y_pred_lr_i):.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_i, y_pred_lr_i, alpha=0.7, color='orchid')
plt.plot([y_test_i.min(), y_test_i.max()], [y_test_i.min(), y_test_i.max()], 'r--')
plt.xlabel("Actual Instagram Usage")
plt.ylabel("Predicted Instagram Usage")
plt.title("Linear Regression: Actual vs Predicted Instagram Usage")
plt.grid(True)
plt.show()

print("---------------------------------------------------------------------")
print()

dt_i = DecisionTreeRegressor(random_state=42)
dt_i.fit(X_train_i, y_train_i)
y_pred_dt_i = dt_i.predict(X_test_i)

print("Instagram Usage - Decision Tree:")
print(f" - RMSE: {np.sqrt(mean_squared_error(y_test_i, y_pred_dt_i)):.2f}")
print(f" - MAE: {mean_absolute_error(y_test_i, y_pred_dt_i):.2f}")
print(f" - R² Score: {r2_score(y_test_i, y_pred_dt_i):.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_i, y_pred_dt_i, alpha=0.7, color='tomato')
plt.plot([y_test_i.min(), y_test_i.max()], [y_test_i.min(), y_test_i.max()], 'r--')
plt.xlabel("Actual Instagram Usage")
plt.ylabel("Predicted Instagram Usage")
plt.title("Decision Tree: Actual vs Predicted Instagram Usage")
plt.grid(True)
plt.show()

print("---------------------------------------------------------------------")
print()

rf_i = RandomForestRegressor(random_state=42)
rf_i.fit(X_train_i, y_train_i)
y_pred_rf_i = rf_i.predict(X_test_i)

print("Instagram Usage - Random Forest:")
print(f" - RMSE: {np.sqrt(mean_squared_error(y_test_i, y_pred_rf_i)):.2f}")
print(f" - MAE: {mean_absolute_error(y_test_i, y_pred_rf_i):.2f}")
print(f" - R² Score: {r2_score(y_test_i, y_pred_rf_i):.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_i, y_pred_rf_i, alpha=0.7, color='forestgreen')
plt.plot([y_test_i.min(), y_test_i.max()], [y_test_i.min(), y_test_i.max()], 'r--')
plt.xlabel("Actual Instagram Usage")
plt.ylabel("Predicted Instagram Usage")
plt.title("Random Forest: Actual vs Predicted Instagram Usage")
plt.grid(True)
plt.show()

# 3. Predicting Reddit Usage
X_reddit = df[features]
y_reddit = df['Reddit Usage']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reddit, y_reddit, test_size=0.2, random_state=42)

lr_r = LinearRegression()
lr_r.fit(X_train_r, y_train_r)
y_pred_lr_r = lr_r.predict(X_test_r)

print("Reddit Usage - Linear Regression:")
print(f" - RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_lr_r)):.2f}")
print(f" - MAE: {mean_absolute_error(y_test_r, y_pred_lr_r):.2f}")
print(f" - R² Score: {r2_score(y_test_r, y_pred_lr_r):.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_r, y_pred_lr_r, alpha=0.7, color='dodgerblue')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Actual Reddit Usage")
plt.ylabel("Predicted Reddit Usage")
plt.title("Linear Regression: Actual vs Predicted Reddit Usage")
plt.grid(True)
plt.show()

print("---------------------------------------------------------------------")
print()

dt_r = DecisionTreeRegressor(random_state=42)
dt_r.fit(X_train_r, y_train_r)
y_pred_dt_r = dt_r.predict(X_test_r)

print("Reddit Usage - Decision Tree:")
print(f" - RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_dt_r)):.2f}")
print(f" - MAE: {mean_absolute_error(y_test_r, y_pred_dt_r):.2f}")
print(f" - R² Score: {r2_score(y_test_r, y_pred_dt_r):.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_r, y_pred_dt_r, alpha=0.7, color='coral')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Actual Reddit Usage")
plt.ylabel("Predicted Reddit Usage")
plt.title("Decision Tree: Actual vs Predicted Reddit Usage")
plt.grid(True)
plt.show()

print("---------------------------------------------------------------------")
print()

rf_r = RandomForestRegressor(random_state=42)
rf_r.fit(X_train_r, y_train_r)
y_pred_rf_r = rf_r.predict(X_test_r)

print("Reddit Usage - Random Forest:")
print(f" - RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_rf_r)):.2f}")
print(f" - MAE: {mean_absolute_error(y_test_r, y_pred_rf_r):.2f}")
print(f" - R² Score: {r2_score(y_test_r, y_pred_rf_r):.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_r, y_pred_rf_r, alpha=0.7, color='mediumseagreen')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Actual Reddit Usage")
plt.ylabel("Predicted Reddit Usage")
plt.title("Random Forest: Actual vs Predicted Reddit Usage")
plt.grid(True)
plt.show()
