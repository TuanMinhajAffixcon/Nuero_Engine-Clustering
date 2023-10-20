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
import itertools


# st.set_page_config(layout="wide")
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

st.title('NEURO ENGINE - With DBSCAN Clustering')

### making a takens for same vectorization

usecols=['Segment Name']
unsupervised_tokens=[]
affix_seg=pd.read_csv('Affixcon_Segmentation.csv',encoding='latin-1',usecols=usecols).dropna()['Segment Name'].tolist()

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
df_master=pd.read_csv('Matched_data_1k.csv',usecols=usecols)
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

#---------------------------------------------------------------------------------------------------------------------

# df_matched_wine=pd.read_csv('../dataset/wine-samples-500.csv').drop(['Flag'],axis=1)
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
        eps=st.text_input('Enter Epsilon values - Radius [Enter float value] ')
        min_samples=st.text_input('Enter Min Samples - Within the Radius including data points ')

    try:
        if eps=='' or min_samples=='':
            st.warning('Please enter values.')
            raise ValueError("Empty Data")
        else:
            fig, ax = plt.subplots(figsize=(18, 16))
            dbscan = DBSCAN(eps=float(eps), min_samples=int(min_samples))
                        # # Create a scatter plot with standardized data
            plt.scatter(df_pca_standardized[0], df_pca_standardized[1], label='Overall', marker='o')
            # plt.scatter(df_pca_wine[['PC1','PC2']]['PC1'], df_pca_wine[['PC1','PC2']]['PC2'], c=cluster_labels, alpha=1,label='Cluster Centers')
            # dbscan = DBSCAN(eps=float(eps), min_samples=int(min_samples))
            cluster_labels = dbscan.fit_predict(df_pca_wine_standardized)
            unique_labels=np.unique(cluster_labels)
            mapping = {label: index for index, label in enumerate(unique_labels)}
            cluster_labels  = np.array([mapping[label] for label in cluster_labels])

            plt.scatter(df_pca_wine_standardized[0], df_pca_wine_standardized[1], label='Wine', marker='x',c=cluster_labels)

            cluster_centers = np.zeros((len(np.unique(cluster_labels)), df_pca_wine_standardized.shape[1]))
            for label in np.unique(cluster_labels):
                # if label != -1:  # Exclude noise points
                    cluster_centers[label] = df_pca_wine_standardized[cluster_labels == label].mean(axis=0)

            # # Mark cluster centers as stars
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', color='red', s=200, label='Cluster Centers')

            # Rest of the code remains the same
            plt.xlabel('Principal Component 1 (PC1)',color='black')
            plt.ylabel('Principal Component 2 (PC2)',color='black')
            plt.title('PCA: 2D Scatter Plot of Principal Components for Two Scenarios (Standardized)',color='black')
            plt.yticks(color='black')
            plt.xticks(color='black')
            plt.legend()
            # plt.xlabel(x_label,color='white')
            # plt.ylabel(y_label,color='white')
            # plt.title(title,color='white')
            # plt.xticks(rotation=90,color='white')


            # # Calculate and display the percentage of data points in each cluster
            for label, center in zip(np.unique(cluster_labels), cluster_centers):
                # if label == -1:
                    # continue  # Skip noise points
                cluster_data = df_pca_wine_standardized[cluster_labels == label]
                # print(cluster_data.shape[0])
                cluster_percentage = (cluster_data.shape[0] / df_pca_wine_standardized.shape[0]) * 100
                label = f'Cluster {label} ({cluster_percentage:.2f}%)'
                plt.annotate(label, center, color='red', fontsize=10, ha='left')
                
            # # plt.grid(True)
            # # plt.show()

            with col1:
                # plt.grid(True)
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                ax.set_facecolor('none')
                boundaries=['top','right','bottom','left']
                for boundary in boundaries:
                    ax.spines[boundary].set_visible(False)
                st.pyplot(plt)

            df_pca_wine_standardized['Cluster'] = cluster_labels


            unique_labels=np.unique(cluster_labels)

            mapping = {label: index for index, label in enumerate(unique_labels)}
            new_cluster_labels  = np.array([mapping[label] for label in cluster_labels])
            unique_labels=np.unique(new_cluster_labels)

            for i in unique_labels:
                cluster_df=df_pca_wine_standardized[df_pca_wine_standardized['Cluster']==i]
                # st.write(f'cluster record {i}: ',len(cluster_df))
                # st.write(cluster_df)

            df_pca_standardized['cluster'] = np.argmin(cdist(df_pca_standardized[[0, 1]], cluster_centers), axis=1)
            df_pca_standardized['distance_to_center'] = np.min(cdist(df_pca_standardized[[0, 1]], cluster_centers), axis=1)
            df_pca_standardized=df_pca_standardized.sort_values('distance_to_center')

            # st.write("df_pca_standardized",df_pca_standardized)

            # # Convert the result to a DataFrame
            result_df = df_pca_standardized[[0, 1, 'cluster', 'distance_to_center']]
            result_df.columns=['x','y','Cluster', 'distance_to_center']
            # # Display or save the result DataFrame as needed
            with col2:
                st.markdown('Considering all the clusters')
                with st.expander("Click to expand distance to each points"):
                    st.write('distance to each points',result_df)
            #     st.write(result_df.sort_values('distance_to_center'))

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


            with col1:
                # st.write(ss(x, df_pca_standardized['cluster']))
                # st.write(df_pca_standardized)

                x=df_pca_wine_standardized.to_numpy()
                dbscan_cluster=dbscan.fit(x)
                test=df_pca_wine_standardized.copy()
                st.write('Custom values Cluster Score is: ',str(round(ss(x, test['Cluster'])*100,2))+' %')
            # st.write(test)

                # start = 0.9
                # end = 1.0
                # step = 0.05
                # eps_values = []
                # current_value = start
                # while current_value <= end:
                #     eps_values.append(current_value)
                #     current_value += step

                if not os.path.exists('best_params_df'+file_name):

                    min_samples_values = np.arange(1,80).tolist()
                    eps_values = np.linspace(0.1, 1.5, num=80)
                    combinations = list(itertools.product(eps_values, min_samples_values))
                    data = []
                    for element in combinations:
                        eps, min_samples = element
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = dbscan.fit_predict(x)
                        score = ss(x, labels)
                        # Replace the following line with your score calculation logic
                        # score = x + y  # Example: Calculate a score as the sum of 'x' and 'y'
                        
                        data.append({'eps': eps, 'min_samples': min_samples, 'score': score})
                    best_params_df = pd.DataFrame(data)
                    best_params_df=best_params_df.sort_values('score',ascending=False)
                    joblib.dump(best_params_df, 'best_params_df'+file_name)

                else:
                    loaded_model_instance=joblib.load('best_params_df'+file_name)
                    st.write(loaded_model_instance)

                # best_score = -1
                # best_eps = None
                # best_min_samples = None

                # # Your data

                # # if not os.path.exists('best_params'+file_name):
                # for eps in eps_values:
                #     for min_samples in min_samples_values:
                #         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                #         labels = dbscan.fit_predict(x)
                #         score = ss(x, labels)
                        
                #         if score > best_score:
                #             best_score = score
                #             best_eps = eps
                #             best_min_samples = min_samples

                # best_params = {
                #         "best_eps": best_eps,
                #         "best_min_samples": best_min_samples,
                #         "best_score": best_score
                #     }
                # st.write(best_params)
                # joblib.dump(best_params, 'best_params'+file_name)

# Save the dictionary containing the variables to a file
                # else:
                # loaded_model_instance=joblib.load('best_params'+file_name)
                # best_eps = loaded_model_instance['best_eps']
                # best_min_samples = loaded_model_instance['best_min_samples']
                # best_score = loaded_model_instance['best_score']
                # st.write("Best eps-Radius",best_eps)
                # st.write("Best min_samples-Within the Radius including data points",best_min_samples)
                # st.write(f"Best values Cluster Score is",best_score*100)



    except ValueError as ve:
        st.error(str(ve))

    # fig, ax = plt.subplots(figsize=(18, 16))

    # # # Create a scatter plot with standardized data
    # plt.scatter(df_pca_standardized[0], df_pca_standardized[1], label='Overall', marker='o')
    # # plt.scatter(df_pca_wine[['PC1','PC2']]['PC1'], df_pca_wine[['PC1','PC2']]['PC2'], c=cluster_labels, alpha=1,label='Cluster Centers')
    # # dbscan = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    # cluster_labels = dbscan.fit_predict(df_pca_wine_standardized)
    # unique_labels=np.unique(cluster_labels)
    # mapping = {label: index for index, label in enumerate(unique_labels)}
    # cluster_labels  = np.array([mapping[label] for label in cluster_labels])

    # plt.scatter(df_pca_wine_standardized[0], df_pca_wine_standardized[1], label='Wine', marker='x',c=cluster_labels)

    # cluster_centers = np.zeros((len(np.unique(cluster_labels)), df_pca_wine_standardized.shape[1]))
    # for label in np.unique(cluster_labels):
    #     # if label != -1:  # Exclude noise points
    #         cluster_centers[label] = df_pca_wine_standardized[cluster_labels == label].mean(axis=0)

    # # # Mark cluster centers as stars
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', color='red', s=200, label='Cluster Centers')

    # # Rest of the code remains the same
    # plt.xlabel('Principal Component 1 (PC1)',color='black')
    # plt.ylabel('Principal Component 2 (PC2)',color='black')
    # plt.title('PCA: 2D Scatter Plot of Principal Components for Two Scenarios (Standardized)',color='black')
    # plt.yticks(color='black')
    # plt.xticks(color='black')
    # plt.legend()
    # # plt.xlabel(x_label,color='white')
    # # plt.ylabel(y_label,color='white')
    # # plt.title(title,color='white')
    # # plt.xticks(rotation=90,color='white')


    # # # Calculate and display the percentage of data points in each cluster
    # for label, center in zip(np.unique(cluster_labels), cluster_centers):
    #     # if label == -1:
    #         # continue  # Skip noise points
    #     cluster_data = df_pca_wine_standardized[cluster_labels == label]
    #     # print(cluster_data.shape[0])
    #     cluster_percentage = (cluster_data.shape[0] / df_pca_wine_standardized.shape[0]) * 100
    #     label = f'Cluster {label} ({cluster_percentage:.2f}%)'
    #     plt.annotate(label, center, color='red', fontsize=10, ha='left')
        
    # # # plt.grid(True)
    # # # plt.show()

    # with col1:
    #     # plt.grid(True)
    #     fig.patch.set_alpha(0.0)
    #     ax.patch.set_alpha(0.0)
    #     ax.set_facecolor('none')
    #     boundaries=['top','right','bottom','left']
    #     for boundary in boundaries:
    #         ax.spines[boundary].set_visible(False)
    #     st.pyplot(plt)

    # df_pca_wine_standardized['Cluster'] = cluster_labels


    # unique_labels=np.unique(cluster_labels)

    # mapping = {label: index for index, label in enumerate(unique_labels)}
    # new_cluster_labels  = np.array([mapping[label] for label in cluster_labels])
    # unique_labels=np.unique(new_cluster_labels)

    # for i in unique_labels:
    #     cluster_df=df_pca_wine_standardized[df_pca_wine_standardized['Cluster']==i]
    #     # st.write(f'cluster record {i}: ',len(cluster_df))
    #     # st.write(cluster_df)

    # df_pca_standardized['cluster'] = np.argmin(cdist(df_pca_standardized[[0, 1]], cluster_centers), axis=1)
    # df_pca_standardized['distance_to_center'] = np.min(cdist(df_pca_standardized[[0, 1]], cluster_centers), axis=1)
    # df_pca_standardized=df_pca_standardized.sort_values('distance_to_center')

    # # st.write("df_pca_standardized",df_pca_standardized)

    # # # Convert the result to a DataFrame
    # result_df = df_pca_standardized[[0, 1, 'cluster', 'distance_to_center']]
    # result_df.columns=['x','y','Cluster', 'distance_to_center']
    # # # Display or save the result DataFrame as needed
    # with col2:
    #     st.markdown('Considering all the clusters')
    #     with st.expander("Click to expand distance to each points"):
    #         st.write('distance to each points',result_df)
    # #     st.write(result_df.sort_values('distance_to_center'))

    # with col2:
    #     st.markdown('Considering cluster by cluster')
    #     for i in range(len(np.unique(cluster_labels))):
    #         group_cluster=result_df[result_df['Cluster']==i].sort_values('distance_to_center')
    #         if not group_cluster.empty:
    #             with st.expander(f"Click to expand distance to grouped cluster points cluster: {i}"):
    #                 cluster_percentage = str(round(len(group_cluster)/len(result_df)*100,2))+' %'
    #                 st.write('Cluster Data Record Percentage: ',cluster_percentage)
    #                 # label = f'Cluster {label} ({cluster_percentage:.2f}%)'
    #                 st.write(f'cluster record {i}: ',len(group_cluster))
    #                 st.write(group_cluster.head())


    # with col1:
    #     # st.write(ss(x, df_pca_standardized['cluster']))
    #     # st.write(df_pca_standardized)

    #     x=df_pca_wine_standardized.to_numpy()
    #     dbscan_cluster=dbscan.fit(x)
    #     test=df_pca_wine_standardized.copy()
    #     st.write('Cluster Score is: ',str(round(ss(x, test['Cluster'])*100,2))+' %')

    # # dbscan_cluster=dbscan.fit(x)
    # # test=df_pca_wine_standardized.copy()
    # # test['cluster']=dbscan_cluster.labels_


    # # # Define a range of values for epsilon and min_samples
    # x=df_pca_wine_standardized[[0,1]].to_numpy()
    # st.write(x)

    # start = 0.1
    # end = 0.3
    # step = 0.01

    # epsilon_values = []
    # current_value = start

    # while current_value <= end:
    #     epsilon_values.append(current_value)
    #     current_value += step
    # min_samples_values = np.arange(1,100).tolist()

    # best_score = -1
    # best_params = {}

    # # # Perform the grid search
    # for epsilon in epsilon_values:
    #     for min_samples in min_samples_values:
    #         dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    #         labels = dbscan.fit_predict(x)  # X is your data
    # # #         # Evaluate the clustering using a metric like silhouette score
    #         # st.write(x)
    #         score = ss(x, labels)
    # #         if score > best_score:
    # #             best_score = score
    # #             best_params = {'eps': epsilon, 'min_samples': min_samples}
    #             # st.write("epsilon:", epsilon)
    #             # st.write("min_samples:", min_samples)
    #             # st.write("score",score)



    # # st.write("Best parameters:", best_params)
    # # st.write("Best silhouette score:", best_score)
    # st.write("min_samples_values:", min_samples_values)
    # st.write("epsilon_values:", epsilon_values)


