import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_id = '1yq6aIqR3sUd1MLrWCTwPDpS6pBFlxpPl'
download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
df = pd.read_csv(download_url)

print("Columns:", df.columns)

x = df[['Latitude', 'Longitude']].values

def elbow(x, n):
    sse = []
    for i in range(1, n+1):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(x)
        sse.append(km.inertia_)
    
    plt.figure(figsize=(12, 9))
    plt.plot(range(1, n+1), sse, marker='o')
    plt.xlabel('Cluster Number')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.grid(True)
    #plt.show()

k = 5
indices = np.random.choice(len(x), k, replace=False)
centroids = x[indices]
print("Randomly selected centroids (Latitude, Longitude):")
print(centroids)

plt.figure(figsize=(10, 8))
plt.title("Latitude & Longitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.scatter(df["Longitude"], df["Latitude"], c="blue", s=10, label='Location')
plt.scatter(centroids[:, 1], centroids[:, 0], marker='*', s=200, c='red', label='Centroid')
plt.legend(loc='best')
plt.grid(True)
plt.show()

elbow(x, 25)