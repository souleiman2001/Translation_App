import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = 'C:/Users/souleiman/Downloads/data-final632.csv'
data = pd.read_csv(data_path)

# Selection of the score columns
scores_data = data[['bleu_score', 'cosine_similarity', 'lsa_score']]

# Data standardization
scaler = StandardScaler()
scores_scaled = scaler.fit_transform(scores_data)

# Reduction into 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scores_scaled)

# Creation of a DataFrame for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Combination of the principal components with the tool names
final_df = pd.concat([pca_df, data[['tool_name']]], axis=1)

# Visualization of the results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='tool_name', data=final_df, palette='viridis', s=100)
plt.title('PCA of Translation Tool Scores')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.grid(True)
plt.show()
