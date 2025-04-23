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

# HISTOGRAM VISUALIZATIONS
print("\n" + "="*50)
print("HISTOGRAM VISUALIZATIONS")
print("="*50)

# Histograms for numerical columns (excluding Homework/Project)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('Homework/Project') if 'Homework/Project' in numerical_cols else numerical_cols

# Calculate number of rows needed (3 plots per row)
num_cols = len(numerical_cols)
num_rows = (num_cols + 2) // 3  # Ceiling division

# Create histograms
plt.figure(figsize=(15, 5*num_rows))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(num_rows, 3, i)
    if col in ['School Time', 'Travel Time', 'Sleep Time']:
        # Create temporary dataframe with hours
        temp_data = pd.DataFrame({col: df[col]/60})
        sns.histplot(data=temp_data, x=col)
        plt.xlabel(f'{col} (Hours)')
    else:
        sns.histplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# BOXPLOT VISUALIZATIONS
print("\n" + "="*50)
print("BOXPLOT VISUALIZATIONS")
print("="*50)

# Box plots for numerical columns (excluding Homework/Project)
plt.figure(figsize=(15, 5*num_rows))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(num_rows, 3, i)
    if col in ['School Time', 'Travel Time', 'Sleep Time']:
        # Create temporary dataframe with hours
        temp_data = pd.DataFrame({col: df[col]/60})
        sns.boxplot(data=temp_data, y=col)
        plt.ylabel(f'{col} (Hours)')
    else:
        sns.boxplot(data=df, y=col)
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

# Compare Reddit vs Instagram usage
plt.figure(figsize=(10, 6))
plt.scatter(df['School Time'], df['Reddit Usage'])
plt.xlabel('School Time')
plt.ylabel('Reddit Usage')
plt.title('School Time vs Reddit Usage')
plt.show()

# Relationship between Sleep Time and Sleep Quality
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=df['School Time']/60, y='Instagram Usage')
plt.xlabel('School Time (Hours)')
plt.title('School Time vs Instagram Usage')
plt.show()

# How school time affects sleep quality
plt.figure(figsize=(10, 6))
sns.regplot(x=df['School Time']/60, y=df['Sleep Quality'], data=df)
plt.xlabel('School Time (Hours)')
plt.title('Impact of School Time on Sleep Quality')
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

