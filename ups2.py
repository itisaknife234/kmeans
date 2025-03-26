import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium as gm
from copy import deepcopy
import sys

def main():
    file_id = '1yq6aIqR3sUd1MLrWCTwPDpS6pBFlxpPl'
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

    df = pd.read_csv(download_url)

    if 'Longtitude' in df.columns:
        df.rename(columns={'Longtitude': 'Longitude'}, inplace=True)

    required_cols = {'Latitude', 'Longitude'}
    if not required_cols.issubset(df.columns):
        print(f"에러: CSV에 {required_cols} 열이 모두 존재해야 합니다.")
        sys.exit(1)

    x = df[['Latitude', 'Longitude']].values

    def dist_func(A, B):
        return np.sqrt(np.sum((A - B)**2))

    k = 5
    indices = np.random.choice(len(x), k, replace=False) 
    centroids = x[indices]                             
    clusters = np.zeros(len(x))                          
    C_old = np.zeros_like(centroids)                      
    flag = dist_func(centroids, C_old)                    

    while flag != 0:
        for i in range(len(x)):
            distances = [dist_func(x[i], centroids[j]) for j in range(k)]
            clusters[i] = np.argmin(distances)

        C_old = deepcopy(centroids)

        for i in range(k):
            points_in_cluster = x[clusters == i]
            if len(points_in_cluster) > 0:
                centroids[i][0] = np.mean(points_in_cluster[:, 0])  # Latitude 평균
                centroids[i][1] = np.mean(points_in_cluster[:, 1])  # Longitude 평균

        flag = dist_func(centroids, C_old)

    plt.figure(figsize=(8,6))
    plt.title("Final Clustering Result")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.scatter(x[clusters == 0, 1], x[clusters == 0, 0], s=50, c='red',    marker='o', edgecolor='black', label='A')
    plt.scatter(x[clusters == 1, 1], x[clusters == 1, 0], s=50, c='yellow', marker='x', edgecolor='black', label='B')
    plt.scatter(x[clusters == 2, 1], x[clusters == 2, 0], s=50, c='green',  marker='^', edgecolor='black', label='C')
    plt.scatter(x[clusters == 3, 1], x[clusters == 3, 0], s=50, c='pink',   marker='^', edgecolor='black', label='D')
    plt.scatter(x[clusters == 4, 1], x[clusters == 4, 0], s=50, c='blue',   marker='^', edgecolor='black', label='E')

    plt.scatter(centroids[:, 1], centroids[:, 0], s=250, marker='*', c='black',
                edgecolor='black', label='Final Centroids')

    plt.legend()
    plt.grid()
    plt.savefig("cluster_result.png") 
    print("군집화 결과 그래프를 'cluster_result.png'로 저장했습니다.")
    # plt.show()

    g_map = gm.Map(location=[37.428531, 126.596539], zoom_start=12)

    m_color = ['blue', 'red', 'green', 'purple', 'orange']

    for i in range(k):
        points_in_cluster = x[clusters == i]
        for j in range(len(points_in_cluster)):
            lat = points_in_cluster[j, 0]
            lon = points_in_cluster[j, 1]
            gm.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=m_color[i],
                fill=True,
                fill_color=m_color[i],
                fill_opacity=0.7
            ).add_to(g_map)

    g_map.save("map.html")
    print("지도 결과를 'map.html'로 저장했습니다. 브라우저로 열어보세요.")

if __name__ == "__main__":
    main()
