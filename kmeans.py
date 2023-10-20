import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score as ss
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title='NEURO ENGINE',page_icon=':man_and_woman_holding_hands:',layout='wide')
custom_css = """
<style>
body {
    background-color: #22222E; 
    secondary-background {
    background-color: #FA55AD; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)
st.title('NEURO ENGINE  - With Hierarchical-Agglomerative Clustering')

### making a takens for same vectorization

usecols=['Segment Name']
unsupervised_tokens=[]
affix_seg=pd.read_csv('../dataset/Affixcon_Segmentation.csv',encoding='latin-1',usecols=usecols).dropna()['Segment Name'].tolist()

income=["Under $20,799","$20,800 - $41,599","$41,600 - $64,999","$65,000 - $77,999","$78,000 - $103,999","$104,000 - $155,999","$156,000+"]
age=["<20","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84",">84"]
gender=['Female','Male']
features=[affix_seg,income,age,gender]
for item in features:
    unsupervised_tokens.extend(item)
#-----------------------------------------------------------------------------------------------------------------------------

def vectorizer(ds,vocabulary):
    vectorizer_list=[]
    for sentence in ds['Concatenated']:
        sentence_lst=np.zeros(len(vocabulary))
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split('|'):
                sentence_lst[i]=1
        vectorizer_list.append(sentence_lst)
    return vectorizer_list

usecols=['Age_Range','Gender','Income','interests', 'brands_visited', 'place_categories', 'geobehaviour']
df_master=pd.read_csv('../dataset/Matched_data_1k.csv',usecols=usecols)
df_master['Income']=df_master['Income'].fillna(df_master['Income'].mode()[0])
df_master['Age_Range']=df_master['Age_Range'].fillna(df_master['Age_Range'].mode()[0])
df_master['Gender']=df_master['Gender'].fillna(df_master['Gender'].mode()[0])
df_master=df_master.fillna("")
df_master['Concatenated'] = df_master[['interests', 'brands_visited', 'place_categories','geobehaviour', 'Income', 'Age_Range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)

# vectorized_inputs_master=vectorizer(df_master,unsupervised_tokens)
vectorized_inputs_master = joblib.load('vectorized_inputs_master.pkl')
# joblib.dump(vectorized_inputs_master, 'vectorized_inputs_master.pkl')
scaler = StandardScaler()
df_master_vectorized=scaler.fit_transform(vectorized_inputs_master)

pca = PCA(n_components=2)
components = pca.fit_transform(df_master_vectorized)
df_master_vectorized_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])


# joblib.dump(pca, 'pca_model.pkl')
# joblib.dump(df_master_vectorized_pca, 'df_master_vectorized_pca.pkl')

loaded_pca = joblib.load('pca_model.pkl')
df_master_vectorized_pca = joblib.load('df_master_vectorized_pca.pkl')

col1,col2=st.columns((0.75,0.25))

# plt.scatter(df_master_vectorized_pca['PC1'], df_master_vectorized_pca['PC2'])
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('Master Data Points')
# with col1:
#     st.pyplot(plt)

with col1:
    file_uploader = st.file_uploader(" :file_folder: Upload a file", type=["csv"])
    file_name = file_uploader.name.rsplit(".", 1)[0]
    if not file_name.endswith(".pkl"):
        # Append the .pkl extension if it's not present
        file_name += ".pkl"

if file_uploader is not None:
    df_matched_wine=pd.read_csv(file_uploader).drop(['Flag'],axis=1)
    df_matched_wine.drop('maid',axis=1,inplace=True)
    df_matched_wine['Income']=df_matched_wine['Income'].fillna(df_matched_wine['Income'].mode()[0])
    df_matched_wine['age_range']=df_matched_wine['age_range'].fillna(df_matched_wine['age_range'].mode()[0])
    df_matched_wine['Gender']=df_matched_wine['Gender'].fillna(df_matched_wine['Gender'].mode()[0])
    df_matched_wine=df_matched_wine.fillna("")
    df_matched_wine['Concatenated'] = df_matched_wine[['interests', 'brands_visited', 'place_categories','geobehaviour','Income', 'age_range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)

    if not os.path.exists('vectorized_inputs_wine_'+file_name):
        transformed_data = vectorizer(df_matched_wine,unsupervised_tokens)
        joblib.dump(transformed_data,'vectorized_inputs_wine_'+file_name)
        st.write("save data: Run Again ")
        rerun_button = st.button("Rerun App")

        if rerun_button:
            st.experimental_rerun()

    else:
        vectorized_inputs_wine = joblib.load('vectorized_inputs_wine_'+file_name)
        # st.write("Loaded data:", vectorized_inputs_wine)
        df_wine=scaler.fit_transform(vectorized_inputs_wine)
    #     # df_wine = pd.DataFrame(vectorized_inputs_wine)
        components = pca.fit_transform(df_wine)

        # components=loaded_pca.transform(df_wine)
        df_pca_wine1 = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        if not os.path.exists('df_pca_wine_'+file_name):
            joblib.dump(df_pca_wine1,'df_pca_wine_'+file_name)
            st.write("save data: Run Again ")
            rerun_button = st.button("Rerun App")
        else:
            df_pca_wine = joblib.load('df_pca_wine_'+file_name)

# st.write(df_pca_wine)

# # # plt.figure(figsize=(8, 6))
    plt.scatter(df_pca_wine['PC1'], df_pca_wine['PC2'], marker='*', color='orange', label='Random Data Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Random Data Points')
    # st.pyplot(plt)

df_pca_standardized = pd.DataFrame(scaler.fit_transform(df_master_vectorized_pca))
df_pca_wine_standardized = pd.DataFrame(scaler.fit_transform(df_pca_wine[['PC1','PC2']]))


with col1:
    n_clusters=st.text_input('Enter n_clusters ')

x=df_pca_wine_standardized[[0,1]].to_numpy()

# Assuming you have your data in the variable X and you want to try different numbers of clusters
max_clusters = 100  

best_silhouette_score = -1
best_n_clusters = 2
data = []

for clusters in range(2, max_clusters + 1):
    # st.write(clusters)
    # Create an Agglomerative Clustering model
    agg_clustering = AgglomerativeClustering(n_clusters=clusters, affinity='cosine', linkage='complete')

    # Fit the model and predict cluster labels
    cluster_labels = agg_clustering.fit_predict(x)

    # Calculate the silhouette score
    silhouette_avg = ss(x, cluster_labels, metric='cosine')
    data.append({'cluser': clusters, 'silhouette_avg': silhouette_avg})


data_score = pd.DataFrame(data).sort_values('silhouette_avg',ascending=False)
st.write(data_score.head())



fig, ax = plt.subplots(figsize=(18, 16))
agg_clustering = AgglomerativeClustering(n_clusters=int(n_clusters), affinity='cosine', linkage='complete')
cluster_labels = agg_clustering.fit_predict(df_pca_wine_standardized)

plt.scatter(df_pca_standardized[0], df_pca_standardized[1], label='Overall', marker='o')
plt.scatter(df_pca_wine_standardized[0], df_pca_wine_standardized[1], label='Wine', marker='x', c=cluster_labels)

cluster_centers = np.array([df_pca_wine_standardized[cluster_labels == label].mean(axis=0) for label in np.unique(cluster_labels)])


plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', color='red', s=200, label='Cluster Centers')

plt.xlabel('Principal Component 1 (PC1)', color='black')
plt.ylabel('Principal Component 2 (PC2)', color='black')
plt.title('PCA: 2D Scatter Plot of Principal Components with Agglomerative Clustering and Cluster Centers', color='black')
plt.yticks(color='black')
plt.xticks(color='black')
plt.legend()

for label, center in zip(np.unique(cluster_labels), cluster_centers):
    cluster_data = df_pca_wine_standardized[cluster_labels == label]
    cluster_percentage = (cluster_data.shape[0] / df_pca_wine_standardized.shape[0]) * 100
    label = f'Cluster {label} ({cluster_percentage:.2f}%)'
    plt.annotate(label, center, color='red', fontsize=10, ha='left')

with col1:
    st.pyplot(plt)
df_pca_wine_standardized['Cluster'] = cluster_labels


unique_labels=np.unique(cluster_labels)

mapping = {label: index for index, label in enumerate(unique_labels)}
new_cluster_labels  = np.array([mapping[label] for label in cluster_labels])
unique_labels=np.unique(new_cluster_labels)

for i in unique_labels:
    cluster_df=df_pca_wine_standardized[df_pca_wine_standardized['Cluster']==i]

df_pca_standardized['cluster'] = np.argmin(cdist(df_pca_standardized[[0, 1]], cluster_centers), axis=1)
df_pca_standardized['distance_to_center'] = np.min(cdist(df_pca_standardized[[0, 1]], cluster_centers), axis=1)
df_pca_standardized=df_pca_standardized.sort_values('distance_to_center')


result_df = df_pca_standardized[[0, 1, 'cluster', 'distance_to_center']]
result_df.columns=['x','y','Cluster', 'distance_to_center']

with col2:
    st.markdown('Considering all the clusters')
    with st.expander("Click to expand distance to each points"):
        st.write('distance to each points',result_df)

with col2:
    st.markdown('Considering cluster by cluster')
    for i in range(len(np.unique(cluster_labels))):
        group_cluster=result_df[result_df['Cluster']==i].sort_values('distance_to_center')
        if not group_cluster.empty:
            with st.expander(f"Click to expand distance to grouped cluster points cluster: {i}"):
                cluster_percentage = str(round(len(group_cluster)/len(result_df)*100,2))+' %'
                st.write('Cluster Data Record Percentage: ',cluster_percentage)
                # label = f'Cluster {label} ({cluster_percentage:.2f}%)'
                st.write(f'cluster record {i}: ',len(group_cluster))
                st.write(group_cluster.head())

