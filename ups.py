import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_id = '1yq6aIqR3sUd1MLrWCTwPDpS6pBFlxpPl'
download_url = f'https://drive.google.com/uc?id={file_id}'
df = pd.read_csv(download_url)

print("Shape:", df.shape)
print(df.info())

xy = np.array(df)
x = xy[:, 1:]

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
    plt.show()

elbow(x, 25)
