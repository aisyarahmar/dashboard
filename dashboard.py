import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans

sns.set(style='dark')
st.title("Proyek Analisis Data [Bike-Sharing]\n")

with st.sidebar:
    st.write('''
Nama: Aisya Rahma Rabbania \n
Email: aisyarahmar06@gmail.com \n
ID Dicoding: aisyarahmar''')

#tabel keterangan
season_data = {'Season': [1, 2, 3, 4],
        'Keterangan': ['Spring', 'Summer', 'Fall', 'Winter']}
weathersit_data = {"Weathersit" : [1, 2, 3, 4],
"Keterangan": ["Clear, Few clouds, Partly cloudy, Partly cloudy", "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist", "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", "WinHeavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fogter"]}
with st.sidebar:
    st.write("Keterangan Season")
    df1 = pd.DataFrame(season_data)
    st.sidebar.table(df1)
    st.write("Keterangan Weathersit")
    df2 = pd.DataFrame(weathersit_data)
    st.sidebar.table(df2)  
    
  
# load data
df_hour = pd.read_csv("hour.csv")

#Pertanyaan 1
st.subheader("Faktor Lingkungan yang Mempengaruhi Jumlah Penyewaan Sepeda")

#Correlation matrix
correlation_matrix = df_hour[['season', 'holiday', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
#heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

#Korelasi terendah-tertinggi
highest_corr_vars = correlation_matrix['cnt'].sort_values(ascending=False).index[1:]
lowest_corr_vars = correlation_matrix['cnt'].sort_values().index[:-1]
#barplot
st.write("Korelasi Antara Faktor Lingkungan dengan Jumlah Penyewaan Sepeda")
fig, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(x=correlation_matrix['cnt'][highest_corr_vars].values, y=highest_corr_vars, palette='viridis', ax=ax1)
ax1.set_title('Korelasi Tertinggi dengan Jumlah Penyewaan Sepeda')
ax1.set_xlabel('Korelasi')
ax1.set_ylabel('Variabel')
st.pyplot(fig)

#Clustering
X = df_hour[['season', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_hour['Cluster'] = kmeans.fit_predict(X_scaled)
cluster_stats = df_hour.groupby('Cluster')[['season', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']].mean()
st.write("Statistik untuk Setiap Cluster:")
# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
cluster_stats.plot(kind='bar', ax=ax)
ax.set_title('Statistik Rata-rata untuk Setiap Cluster')
ax.set_xlabel('Cluster')
ax.set_ylabel('Rata-rata')
st.pyplot(fig)


st.write("--------------------------------------------------------------")
#Pertanyaan 2
st.subheader("Waktu Paling Banyak Terjadi Penyewaan Sepeda (dalam Jam)")

#Line chart
hour_segmentation = df_hour.groupby('hr')['cnt'].mean()
st.line_chart(hour_segmentation)


