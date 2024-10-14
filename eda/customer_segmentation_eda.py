import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./data/customer_data.csv')

# Display basic info
print(df.info())

# Display summary statistics
print(df.describe())

# Plot Age distribution
plt.figure(figsize=(8,6))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Plot Income vs Spending Score
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.title('Income vs Spending Score')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
