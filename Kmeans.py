import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load the employee data from the CSV
df = pd.read_csv("iris.csv")

# Step 2: Preprocess the data (encoding categorical variables like 'Name' and 'Project working on')
# We'll use LabelEncoder to convert categorical data into numerical form
label_encoder = LabelEncoder()

df['species'] = label_encoder.fit_transform(df['species'])
#df['Project working on'] = label_encoder.fit_transform(df['Project working on'])

# Step 3: Select features for clustering (independent variables)
# We can use 'Emp_ID', 'Name', 'Dept_ID', and 'Project working on' as features
X = df[['sepal_length', 'sepal_width', 'petal_length', 'species']]

# Optional: Standardize the features to bring them to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # We choose 2 clusters arbitrarily
kmeans.fit(X_scaled)

# Step 5: Add the cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Print the dataframe to see the clusters
print("\iris data with Clusters:\n", df)

# Step 6: Visualize the clusters (using first 2 features for 2D plotting)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("K-Means Clustering (2 Clusters)")
plt.xlabel('sepal_length (Standardized)')
plt.ylabel('species (Standardized)')
plt.show()
print(plt.scatter())
plt.show()
