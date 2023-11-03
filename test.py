import streamlit as st
import pandas as pd

file_uploader = st.file_uploader(" :file_folder: Upload a file", type=["csv"])

if file_uploader is not None:
    df_matched_wine=pd.read_csv(file_uploader).drop(['Flag'],axis=1)
    df_matched_wine.drop('maid',axis=1,inplace=True)
    df_matched_wine['Income']=df_matched_wine['Income'].fillna(df_matched_wine['Income'].mode()[0])
    df_matched_wine['age_range']=df_matched_wine['age_range'].fillna(df_matched_wine['age_range'].mode()[0])
    df_matched_wine['Gender']=df_matched_wine['Gender'].fillna(df_matched_wine['Gender'].mode()[0])
    df_matched_wine=df_matched_wine.fillna("")
    df_matched_wine['Concatenated'] = df_matched_wine[['interests', 'brands_visited', 'place_categories','geobehaviour','Income', 'age_range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)
    
    income_percentages_sample = round(df_matched_wine['Income'].value_counts(normalize=True) * 100,2)
    custom_order_income = ['unknown_income', "Under $20,799", "$20,800 - $41,599", "$41,600 - $64,999","$65,000 - $77,999","$78,000 - $103,999","$104,000 - $155,999","$156,000+"]
    filtered_custom_order_income = [item for item in custom_order_income if item in income_percentages_sample.index]
    income_percentages_sample = income_percentages_sample.reindex(filtered_custom_order_income)

    gender_percentages_sample = round(df_matched_wine['Gender'].value_counts(normalize=True) * 100,2)
    custom_order_gender = ['unknown_gender',"Male","Female"]
    filtered_custom_order_gender = [item for item in custom_order_gender if item in gender_percentages_sample.index]
    gender_percentages_sample = gender_percentages_sample.reindex(filtered_custom_order_gender)

    age_percentages_sample = round(df_matched_wine['age_range'].value_counts(normalize=True) * 100,2)
    custom_order_age = ['unknown_age',"<20","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84",">84"]
    filtered_custom_order_age = [item for item in custom_order_age if item in age_percentages_sample.index]
    age_percentages_sample = age_percentages_sample.reindex(filtered_custom_order_age)



    st.write(income_percentages_sample)
    st.write(gender_percentages_sample)
    st.write(age_percentages_sample)
