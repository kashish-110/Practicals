import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset (with 4 features and 8 samples)
data = {
    'Feature1': [2, 4, 5, 10, 12, 14],
    'Feature2': [1, 3, 4, 9, 11, 13],
    
}
df = pd.DataFrame(data)

print("Original Data:\n", df)

# Step 2: Standardize the dataset (PCA requires data to be on the same scale)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 3: Apply PCA to reduce dimensions (we will reduce to 2 principal components for visualization)
pca = PCA(n_components=2)  # Reduce to 2 dimensions
df_pca = pca.fit_transform(df_scaled)

# Step 4: Create a new DataFrame with the principal components
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

print("\nData after PCA (2 Principal Components):\n", df_pca)

# Step 5: Visualize the data after PCA
plt.scatter(df_pca['PC1'], df_pca['PC2'], c='blue', marker='o')
plt.title("PCA Result: 2 Principal Components")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
