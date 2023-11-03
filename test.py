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
from sklearn.neighbors import NearestNeighbors
import math
import plotly.express as px
from collections import Counter

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
df_seg=pd.read_csv('affixcon_segments.csv',encoding='latin-1').dropna(subset=['segment_name'])
df_seg.category = df_seg.category.str.upper()
df_seg.segment_name = df_seg.segment_name.str.title()
df_seg = df_seg[df_seg['category'] != 'APP CATEGORY']
affix_seg=pd.read_csv('Affixcon_Segmentation.csv',encoding='latin-1',usecols=usecols).dropna()['Segment Name'].tolist()


income=["Under $20,799","$20,800 - $41,599","$41,600 - $64,999","$65,000 - $77,999","$78,000 - $103,999","$104,000 - $155,999","$156,000+"]
age=["<20","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84",">84"]
gender=['Female','Male']
features=[affix_seg,income,age,gender]
for item in features:
    unsupervised_tokens.extend(item)
# st.write(unsupervised_tokens)

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

# methodology=st.checkbox('Select Filter Segment for Master Data')


# usecols=['Age_Range','Gender','Income','interests', 'brands_visited', 'place_categories', 'geobehaviour']
# df_master=pd.read_csv('Matched_data_1k.csv',usecols=usecols)
# df_master['Income']=df_master['Income'].fillna(df_master['Income'].mode()[0])
# df_master['Age_Range']=df_master['Age_Range'].fillna(df_master['Age_Range'].mode()[0])
# df_master['Gender']=df_master['Gender'].fillna(df_master['Gender'].mode()[0])
# df_master=df_master.fillna("")
# selected_columns = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']



# if methodology:
#     option = st.selectbox("Select inputs", ('industry', 'segments', 'code'))
#     if option=='industry':
#             industry_list = df_seg['industry'].dropna().unique().tolist()
#             selected_industry = st.selectbox(' :bookmark_tabs: Enter Industry:', industry_list)

#             segment_industry_dict = df_seg.groupby('industry')['segment_name'].apply(list).to_dict()
#             item_list = segment_industry_dict[selected_industry]
#             select_all_segments = st.checkbox("Select All Segments",value=True)
#             if select_all_segments:
#                 selected_segments = item_list
#             else:
#                 selected_segments = st.multiselect("Select one or more segments:", item_list)

#     elif option == 'segments':
#         segment_list=df_seg['segment_name'].dropna().unique().tolist()
#         # st.subheader('give segments as comma separated values')
#         selected_segments = st.multiselect(" :bookmark_tabs: Enter segments as comma separated values",segment_list)
#         if len(selected_segments)>0:
#             selected_segments=[selected_segments]

#     elif option == 'code':
#         # st.subheader('give a code')
#         selected_code = st.text_input(":bookmark_tabs: Enter code")
#         item_list = []
#         segment_industry_dict = df_seg.groupby('code')['segment_name'].apply(list).to_dict()
#         def find_similar_codes(input_code, df):
#             similar_codes = []
#             for index, row in df.iterrows():
#                 code = row['code']
#                 if isinstance(code, str) and code.startswith(input_code):
#                     similar_codes.append(code)
#             return similar_codes
        

#         user_contain_list = list(set(find_similar_codes(selected_code, df_seg)))

#         if selected_code in user_contain_list:
#             for code in user_contain_list:
#                 item_list_code = segment_industry_dict[code]
#                 for item in item_list_code:
#                     item_list.append(item)
#         else:
#             item_list = []
#         # Create a checkbox to select all segments
#         select_all_segments = st.checkbox("Select All Segments",value=True)

#         # If the "Select All Segments" checkbox is checked, select all segments
#         if select_all_segments:
#             selected_segments = item_list
#         else:
#             # Create a multiselect widget
#             selected_segments = st.multiselect("Select one or more segments:", item_list)

#     segment_category_dict = df_seg.set_index('segment_name')['category'].to_dict()
#     result_dict = {}
#     filtered_dict = {key: value for key, value in segment_category_dict.items() if key in selected_segments}

#     for key, value in filtered_dict.items():

#         if value not in result_dict:
#             result_dict[value] = []

#         result_dict[value].append(key)
#         result_dict = {key: values for key, values in result_dict.items()}

#     if 'BRAND VISITED' in result_dict and 'BRANDS VISITED' in result_dict:
#         # Extend the 'a' values with 'a1' values
#         result_dict['BRAND VISITED'].extend(result_dict['BRANDS VISITED'])
#         # Delete the 'a1' key
#         del result_dict['BRANDS VISITED']

#     selected_category = st.sidebar.radio("Select one option:", list(result_dict.keys()))
#     if selected_segments:
#         if selected_category == 'INTERESTS':
#             segment_list=result_dict['INTERESTS']
#         elif selected_category == 'BRAND VISITED':
#             segment_list=result_dict['BRAND VISITED']
#         elif selected_category == 'PLACE CATEGORIES':
#             segment_list=result_dict['PLACE CATEGORIES']
#         elif selected_category == 'GEO BEHAVIOUR':
#             segment_list=result_dict['GEO BEHAVIOUR']
#     else:
#         segment_list=[]

    
#     # st.write(segment_list)
#     for j in segment_list:
#         st.sidebar.write(j)

#     def filter_condition(df,lst):
#         filter_conditions = [df[col_name].apply(lambda x: any(item in str(x).split('|') for item in lst))
#             for col_name in selected_columns]
#         final_condition = filter_conditions[0]
#         for condition in filter_conditions[1:]:
#             final_condition = final_condition | condition
#         df_new = df[final_condition]
#         return df_new
#     df_master_filtered=filter_condition(df_master,selected_segments)
#     def filter_items(column):
#         return [item for item in column.split('|') if item in selected_segments]
#     columns_to_filter = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
#     for column in columns_to_filter:
#         df_master_filtered[column] = df_master_filtered[column].apply(filter_items)
#     df_master_filtered[columns_to_filter] = df_master_filtered[columns_to_filter].applymap(lambda x: '|'.join(x))

#     df_master_filtered['Concatenated'] = df_master_filtered[['interests', 'brands_visited', 'place_categories','geobehaviour', 'Income', 'Age_Range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)
#     vectorized_inputs_master_filtered=vectorizer(df_master_filtered,unsupervised_tokens)
#     scaler = StandardScaler()
#     df_master_vectorized_filtered=scaler.fit_transform(vectorized_inputs_master_filtered)

#     pca = PCA(n_components=2)
#     components = pca.fit_transform(df_master_vectorized_filtered)
#     df_master_vectorized_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
#     # st.write(((df_master_vectorized_pca)))

# else:

#     df_master['Concatenated'] = df_master[['interests', 'brands_visited', 'place_categories','geobehaviour', 'Income', 'Age_Range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)


#     # vectorized_inputs_master=vectorizer(df_master,unsupervised_tokens)
#     vectorized_inputs_master = joblib.load('vectorized_inputs_master.pkl')
#     # joblib.dump(vectorized_inputs_master, 'vectorized_inputs_master.pkl')
#     scaler = StandardScaler()
#     df_master_vectorized=scaler.fit_transform(vectorized_inputs_master)

#     pca = PCA(n_components=2)
#     components = pca.fit_transform(df_master_vectorized)
#     df_master_vectorized_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])


#     # joblib.dump(pca, 'pca_model.pkl')
#     # joblib.dump(df_master_vectorized_pca, 'df_master_vectorized_pca.pkl')

#     loaded_pca = joblib.load('pca_model.pkl')
#     df_master_vectorized_pca = joblib.load('df_master_vectorized_pca.pkl')


col1,col2=st.columns((0.75,0.25))



#---------------------------------------------------------------------------------------------------------------------

# df_matched_wine=pd.read_csv('../dataset/wine-samples-500.csv').drop(['Flag'],axis=1)
with col1:
    file_uploader = st.file_uploader(" :file_folder: Upload a file", type=["csv"])
    file_name = file_uploader.name.rsplit(".", 1)[0]
    if not file_name.endswith(".pkl"):
        # Append the .pkl extension if it's not present
        file_name += ".pkl"
# methods=st.radio("Select Method", ('Select All Segments', 'Select Filtered Segments'))
# methodology=st.checkbox('Select Filtered Segments')

if file_uploader is not None:
    df_matched_wine=pd.read_csv(file_uploader).drop(['Flag'],axis=1)
    df_matched_wine.drop('maid',axis=1,inplace=True)
    df_matched_wine['Income']=df_matched_wine['Income'].fillna(df_matched_wine['Income'].mode()[0])
    df_matched_wine['age_range']=df_matched_wine['age_range'].fillna(df_matched_wine['age_range'].mode()[0])
    df_matched_wine['Gender']=df_matched_wine['Gender'].fillna(df_matched_wine['Gender'].mode()[0])
    df_matched_wine=df_matched_wine.fillna("")
    df_matched_wine['Concatenated'] = df_matched_wine[['interests', 'brands_visited', 'place_categories','geobehaviour','Income', 'age_range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)
    def show_demographics(df_matched_wine):
        income_percentages_sample = round(df_matched_wine['Income'].value_counts(normalize=True) * 100,2)
        gender_percentages_sample = round(df_matched_wine['Gender'].value_counts(normalize=True) * 100,2)
        age_percentages_sample = round(df_matched_wine['age_range'].value_counts(normalize=True) * 100,2)
        vocab=Counter()
        for col in ['interests', 'brands_visited', 'place_categories','geobehaviour']:
            for line in df_matched_wine[col]:
                vocab.update(line.split("|"))
        vocab = {key: value for key, value in vocab.items() if key.strip() != ''}
        segments = px.bar(x=list(vocab.keys()), y=list(vocab.values()), title="Category Counts")
        col1,col2,col3,col4=st.columns((4))
        def demographics_sample(dem_df,dem_col):
            dem_df = dem_df.reset_index()
            dem_df.columns = [f'{dem_col} Category', 'Percentage']
            fig = px.bar(dem_df, x='Percentage', y=f'{dem_col} Category', orientation='h', text='Percentage')
            fig.update_layout(title=f'{dem_col} Category', xaxis_title='Percentage (%)', yaxis_title=f'{dem_col} Category')
            return fig

        with col1:
            with st.expander('Show Income'):
                income_sample=demographics_sample(income_percentages_sample,'Income')
                st.plotly_chart(income_sample)
        with col2:
            with st.expander('Show Gender'):
                gender_sample=demographics_sample(gender_percentages_sample,'Gender')
                st.plotly_chart(gender_sample)
        with col3:
            with st.expander('Show Age Groups'):
                age_sample=demographics_sample(age_percentages_sample,'Age')
                st.plotly_chart(age_sample)
        with col4:
            with st.expander('Show Segments'):
                st.plotly_chart(segments)
    show_demographics(df_matched_wine)
    # st.write(income_percentages_sample)
    # st.write(gender_percentages_sample)
    # st.write(age_percentages_sample)

    col1,col2=st.columns((0.75,0.25))

    scaler = StandardScaler()
    pca = PCA(n_components=2)


    if not os.path.exists(f'vectorized_inputs_wine_'+file_name):
        transformed_data = vectorizer(df_matched_wine,unsupervised_tokens)
        joblib.dump(transformed_data,f'vectorized_inputs_wine_'+file_name)
        st.write("save data: Run Again ")
        rerun_button = st.button("Rerun App")

        if rerun_button:
            st.experimental_rerun()

    else:
        vectorized_inputs_wine = joblib.load(f'vectorized_inputs_wine_'+file_name)
        # st.write("Loaded data:", vectorized_inputs_wine)
        df_wine=scaler.fit_transform(vectorized_inputs_wine)
    #     # df_wine = pd.DataFrame(vectorized_inputs_wine)
        components = pca.fit_transform(df_wine)

        # components=loaded_pca.transform(df_wine)
        df_pca_wine1 = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        if not os.path.exists(f'df_pca_wine_'+file_name):
            joblib.dump(df_pca_wine1,f'df_pca_wine_'+file_name)
            st.write("save data: Run Again ")
            rerun_button = st.button("Rerun App")
        else:
            df_pca_wine = joblib.load(f'df_pca_wine_'+file_name)

methodology=st.checkbox('Select Filter Segment for Master Data')

usecols=['age_range','Gender','Income','interests', 'brands_visited', 'place_categories', 'geobehaviour']
df_master=pd.read_csv('Matched_data_1k.csv',usecols=usecols)
df_master['Income']=df_master['Income'].fillna(df_master['Income'].mode()[0])
df_master['age_range']=df_master['age_range'].fillna(df_master['age_range'].mode()[0])
df_master['Gender']=df_master['Gender'].fillna(df_master['Gender'].mode()[0])
df_master=df_master.fillna("")
selected_columns = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
st.write('total Master Data Count is: ',len(df_master))

col1,col2,col3=st.columns((3))
with col1:
    income=st.multiselect('Select Income ',df_master['Income'].unique(), default=df_master['Income'].unique())
with col2:
    gender=st.multiselect('Select Gender ',df_master['Gender'].unique(), default=df_master['Gender'].unique())
with col3:
    age_category=st.multiselect('Select Age ',df_master['age_range'].unique(), default=df_master['age_range'].unique())
df_master=df_master.query('age_range ==@age_category & Gender==@gender & Income==@income')
st.write('Data Record of Filtered Master Table: ',len(df_master))

if methodology:
    option = st.selectbox("Select inputs", ('industry', 'segments', 'code'))
    if option=='industry':
            industry_list = df_seg['industry'].dropna().unique().tolist()
            selected_industry = st.selectbox(' :bookmark_tabs: Enter Industry:', industry_list)

            segment_industry_dict = df_seg.groupby('industry')['segment_name'].apply(list).to_dict()
            item_list = segment_industry_dict[selected_industry]
            select_all_segments = st.checkbox("Select All Segments",value=True)
            if select_all_segments:
                selected_segments = item_list
            else:
                selected_segments = st.multiselect("Select one or more segments:", item_list)

    elif option == 'segments':
        segment_list=df_seg['segment_name'].dropna().unique().tolist()
        # st.subheader('give segments as comma separated values')
        selected_segments = st.multiselect(" :bookmark_tabs: Enter segments as comma separated values",segment_list)
        if len(selected_segments)>0:
            selected_segments=[selected_segments]

    elif option == 'code':
        # st.subheader('give a code')
        selected_code = st.text_input(":bookmark_tabs: Enter code")
        item_list = []
        segment_industry_dict = df_seg.groupby('code')['segment_name'].apply(list).to_dict()
        def find_similar_codes(input_code, df):
            similar_codes = []
            for index, row in df.iterrows():
                code = row['code']
                if isinstance(code, str) and code.startswith(input_code):
                    similar_codes.append(code)
            return similar_codes
        

        user_contain_list = list(set(find_similar_codes(selected_code, df_seg)))

        if selected_code in user_contain_list:
            for code in user_contain_list:
                item_list_code = segment_industry_dict[code]
                for item in item_list_code:
                    item_list.append(item)
        else:
            item_list = []
        # Create a checkbox to select all segments
        select_all_segments = st.checkbox("Select All Segments",value=True)

        # If the "Select All Segments" checkbox is checked, select all segments
        if select_all_segments:
            selected_segments = item_list
        else:
            # Create a multiselect widget
            selected_segments = st.multiselect("Select one or more segments:", item_list)

    segment_category_dict = df_seg.set_index('segment_name')['category'].to_dict()
    result_dict = {}
    filtered_dict = {key: value for key, value in segment_category_dict.items() if key in selected_segments}

    for key, value in filtered_dict.items():

        if value not in result_dict:
            result_dict[value] = []

        result_dict[value].append(key)
        result_dict = {key: values for key, values in result_dict.items()}

    if 'BRAND VISITED' in result_dict and 'BRANDS VISITED' in result_dict:
        # Extend the 'a' values with 'a1' values
        result_dict['BRAND VISITED'].extend(result_dict['BRANDS VISITED'])
        # Delete the 'a1' key
        del result_dict['BRANDS VISITED']

    selected_category = st.sidebar.radio("Select one option:", list(result_dict.keys()))
    if selected_segments:
        if selected_category == 'INTERESTS':
            segment_list=result_dict['INTERESTS']
        elif selected_category == 'BRAND VISITED':
            segment_list=result_dict['BRAND VISITED']
        elif selected_category == 'PLACE CATEGORIES':
            segment_list=result_dict['PLACE CATEGORIES']
        elif selected_category == 'GEO BEHAVIOUR':
            segment_list=result_dict['GEO BEHAVIOUR']
    else:
        segment_list=[]

    
    # st.write(segment_list)
    for j in segment_list:
        st.sidebar.write(j)

    def filter_condition(df,lst):
        filter_conditions = [df[col_name].apply(lambda x: any(item in str(x).split('|') for item in lst))
            for col_name in selected_columns]
        final_condition = filter_conditions[0]
        for condition in filter_conditions[1:]:
            final_condition = final_condition | condition
        df_new = df[final_condition]
        return df_new
    df_master=filter_condition(df_master,selected_segments)
    def filter_items(column):
        return [item for item in column.split('|') if item in selected_segments]
    columns_to_filter = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
    for column in columns_to_filter:
        df_master[column] = df_master[column].apply(filter_items)
    df_master[columns_to_filter] = df_master[columns_to_filter].applymap(lambda x: '|'.join(x))
    
    col1,col2=st.columns((0.75,0.25))

    with col1:
        with st.expander('Show Master Data Table'):
            st.write(df_master)
    st.write('Matching Master Data Count is: ',len(df_master))

    

    df_master['Concatenated'] = df_master[['interests', 'brands_visited', 'place_categories','geobehaviour', 'Income', 'age_range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)
    vectorized_inputs_master_filtered=vectorizer(df_master,unsupervised_tokens)
    scaler = StandardScaler()
    df_master_vectorized_filtered=scaler.fit_transform(vectorized_inputs_master_filtered)

    pca = PCA(n_components=2)
    components = pca.fit_transform(df_master_vectorized_filtered)
    df_master_vectorized_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    # df_master_vectorized_pca['index']=df_master.index.tolist()
    df_master_vectorized_pca = df_master_vectorized_pca.set_index(pd.Index(df_master.index.tolist()))

    # st.write(df_master_vectorized_pca)

else:
    show_demographics(df_master)
    df_master['Concatenated'] = df_master[['interests', 'brands_visited', 'place_categories','geobehaviour', 'Income', 'age_range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)
    col1,col2=st.columns((0.75,0.25))
    # st.write('Matching Master Data Count is: ',len(df_master))
    with col1:
        with st.expander('Show Master Data Table'):
            st.write(df_master)

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

# st.write(df_master)
# index_list=result_df.index.tolist()
# index_list_master=df_master.index.tolist()
# st.write(index_list_master)
# # # plt.figure(figsize=(8, 6))
plt.scatter(df_pca_wine['PC1'], df_pca_wine['PC2'], marker='*', color='orange', label='Random Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of Random Data Points')
# st.pyplot(plt)

df_pca_standardized = pd.DataFrame(scaler.fit_transform(df_master_vectorized_pca))
df_pca_standardized = df_pca_standardized.set_index(pd.Index(df_master.index.tolist()))

df_pca_wine_standardized = pd.DataFrame(scaler.fit_transform(df_pca_wine[['PC1','PC2']]))

# st.write((df_pca_standardized))
k_values = np.arange(1,10).tolist()
k_distances = []  # This list will store the k-distances for each k.

for k in k_values:
    nbrs = NearestNeighbors(n_neighbors=k).fit(df_pca_wine_standardized)
    distances, _ = nbrs.kneighbors(df_pca_wine_standardized)
    k_distances.append(distances[:, -1])

plt.figure()
for i in range(len(k_values)):
    plt.plot(sorted(k_distances[i]), label=f'k={k_values[i]}')
plt.xlabel('Data Points (sorted)')
plt.ylabel('k-distance')
plt.legend()
plt.grid()
show_kdist_graph=st.checkbox('Select for show k-Dist-Graph for best hyper parameter')
col1,col2=st.columns((0.75,0.25))

if show_kdist_graph:
    with col1:
        st.pyplot(plt)

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
                required_data_percentage=st.select_slider('select required percentage from master data',([i for i in range(10, 110, 10)]))
                slicing_data=int(len(result_df)*int(required_data_percentage)/100)
                index_list=result_df.index.tolist()[:slicing_data]
                # st.write(index_list)

                filtered_df = df_master.loc[index_list]
                st.write(filtered_df)
                # def demographics_filtered(dem):
                income_percentages_filtered = round(df_matched_wine['Income'].value_counts(normalize=True) * 100,2)
                income_percentages_sample = round(filtered_df['Income'].value_counts(normalize=True) * 100,2)
                filtered_df_income = pd.DataFrame({'Income Percentages': income_percentages_filtered.index, 'Percentage': income_percentages_filtered.values})
                sample_df_income = pd.DataFrame({'Income Percentages': income_percentages_sample.index, 'Percentage': income_percentages_sample.values})
                combined_df = pd.concat([filtered_df_income, sample_df_income], axis=0, keys=['Filtered Data', 'Sample Data'], names=['Data Type'])
                fig = px.bar(combined_df, x='Percentage', y='Income Percentages', text='Percentage', title='Income Percentages - Filtered Data and Sample Data', color=combined_df.index.get_level_values(0))
                st.plotly_chart(fig)
                

                Gender_percentages_filtered = round(df_matched_wine['Gender'].value_counts(normalize=True) * 100,2)
                Gender_percentages_sample = round(filtered_df['Gender'].value_counts(normalize=True) * 100,2)
                filtered_df_Gender = pd.DataFrame({'Gender Percentages': Gender_percentages_filtered.index, 'Percentage': Gender_percentages_filtered.values})
                sample_df_Gender = pd.DataFrame({'Gender Percentages': Gender_percentages_sample.index, 'Percentage': Gender_percentages_sample.values})
                combined_df = pd.concat([filtered_df_Gender, sample_df_Gender], axis=0, keys=['Filtered Data', 'Sample Data'], names=['Data Type'])
                fig = px.bar(combined_df, x='Percentage', y='Gender Percentages', text='Percentage', title='Gender Percentages - Filtered Data and Sample Data', color=combined_df.index.get_level_values(0))
                st.plotly_chart(fig)

                Age_range_percentages_filtered = round(df_matched_wine['age_range'].value_counts(normalize=True) * 100,2)
                Age_range_percentages_sample = round(filtered_df['age_range'].value_counts(normalize=True) * 100,2)
                filtered_df_age_range = pd.DataFrame({'age_range Percentages': Age_range_percentages_filtered.index, 'Percentage': Age_range_percentages_filtered.values})
                sample_df_age_range = pd.DataFrame({'age_range Percentages': Age_range_percentages_sample.index, 'Percentage': Age_range_percentages_sample.values})
                combined_df = pd.concat([filtered_df_age_range, sample_df_age_range], axis=0, keys=['Filtered Data', 'Sample Data'], names=['Data Type'])
                fig = px.bar(combined_df, x='Percentage', y='age_range Percentages', text='Percentage', title='age_range Percentages - Filtered Data and Sample Data', color=combined_df.index.get_level_values(0))
                st.plotly_chart(fig)
                # st.write(combined_df)



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


            x=df_pca_wine_standardized.to_numpy()
            dbscan_cluster=dbscan.fit(x)
            test=df_pca_wine_standardized.copy()
            st.write('Custom values Cluster Score is: ',str(round(ss(x, test['Cluster'])*100,2))+' %')
        # st.write(test)


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
                with col2:
                    st.write(loaded_model_instance)




except ValueError as ve:
    st.error(str(ve))




