# Week-7

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Task 1: Load and Explore the Dataset
print("=" * 50)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("=" * 50)

try:
    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {iris_df.shape}")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(iris_df.head())
    
    # Explore data types
    print("\nData types:")
    print(iris_df.dtypes)
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(iris_df.isnull().sum())
    
    # Since Iris dataset has no missing values, we'll just confirm that
    print("\nNo missing values found in this dataset.")
    
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\n" + "=" * 50)
print("TASK 2: BASIC DATA ANALYSIS")
print("=" * 50)

# Basic statistics
print("Basic statistics of numerical columns:")
print(iris_df.describe())

# Group by species and compute mean of numerical columns
print("\nMean values by species:")
species_means = iris_df.groupby('species').mean()
print(species_means)

# Additional analysis: find the species with the largest petals
max_petal_length = iris_df.groupby('species')['petal length (cm)'].max()
print(f"\nMaximum petal length by species:\n{max_petal_length}")

# Task 3: Data Visualization
print("\n" + "=" * 50)
print("TASK 3: DATA VISUALIZATION")
print("=" * 50)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')

# 1. Line chart (using index as pseudo-time series)
axes[0, 0].plot(iris_df.index, iris_df['sepal length (cm)'], 
                color='royalblue', linewidth=1, alpha=0.7)
axes[0, 0].set_title('Sepal Length Trend (by index)', fontweight='bold')
axes[0, 0].set_xlabel('Observation Index')
axes[0, 0].set_ylabel('Sepal Length (cm)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart - average petal length per species
species_means['petal length (cm)'].plot(kind='bar', ax=axes[0, 1], 
                                       color=['lightcoral', 'lightgreen', 'lightblue'])
axes[0, 1].set_title('Average Petal Length by Species', fontweight='bold')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Petal Length (cm)')
axes[0, 1].tick_params(axis='x', rotation=0)

# 3. Histogram - distribution of sepal width
axes[1, 0].hist(iris_df['sepal width (cm)'], bins=15, color='lightseagreen', 
                edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribution of Sepal Width', fontweight='bold')
axes[1, 0].set_xlabel('Sepal Width (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter plot - sepal length vs petal length
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species in iris_df['species'].unique():
    species_data = iris_df[iris_df['species'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], 
                      species_data['petal length (cm)'], 
                      label=species, alpha=0.7, s=50)
axes[1, 1].set_title('Sepal Length vs Petal Length', fontweight='bold')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional visualizations
# Box plot to show distribution of features by species
plt.figure(figsize=(12, 8))
iris_df.boxplot(by='species', column=['sepal length (cm)', 'sepal width (cm)', 
                                     'petal length (cm)', 'petal width (cm)'])
plt.suptitle('Feature Distributions by Species')
plt.savefig('iris_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = iris_df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Iris Features')
plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Pairplot to show relationships between all variables
sns.pairplot(iris_df, hue='species', diag_kind='hist', palette='husl')
plt.suptitle('Pairplot of Iris Dataset by Species', y=1.02)
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Findings and Observations
print("\n" + "=" * 50)
print("FINDINGS AND OBSERVATIONS")
print("=" * 50)
print("1. Setosa species has distinctly smaller petals compared to the other two species.")
print("2. Virginica has the longest sepals and petals on average.")
print("3. Sepal width has the most normal distribution among all features.")
print("4. There is a strong positive correlation between petal length and petal width.")
print("5. Setosa is clearly separable from the other two species based on petal measurements.")
print("6. Versicolor and Virginica have some overlap but are generally separable.")
print("7. The distribution of sepal width is approximately normal with a mean around 3 cm.")
print("8. The relationship between sepal length and petal length shows clear clustering by species.")

# Save the analysis to a CSV file for reference
iris_df.to_csv('iris_analysis_data.csv', index=False)
print("\nAnalysis complete. Results saved to CSV and image files.")
