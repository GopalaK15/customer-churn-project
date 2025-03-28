import pickle
import streamlit as st
import pandas as pd
from PIL import Image

with open('XGBoost_model.pkl', 'rb') as f:
    dv, xgb_model = pickle.load(f)


def main():
    add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.title("Predicting Customer Churn")
    if add_selectbox == 'Online':
        Gender = st.selectbox('Gender:', ['male', 'female'])
        ContractLength = st.selectbox(' Contract Length:', ['Annual', 'Monthly', 'Quarterly'])
        SubscriptionType = st.selectbox(' Subscription Type:', ['Basic', 'Premium', 'Standard'])
        Age =st.number_input(' Age of the customer :', min_value=0, max_value=100, value=0)
        LastInteraction =st.number_input(' Last Interaction in minutes :', min_value=0, max_value=1000, value=0)
        TotalSpend =st.number_input(' Total Time Spent in minutes :', min_value=0, max_value=10000, value=0)
        PaymentDelay = st.number_input('Payment Delay in Days :', min_value=0, max_value=100, value=0)
        Tenure = st.number_input('Tenure in months :', min_value=0, max_value=300, value=0)
        UsageFrequency = st.number_input('Usage Frequency :', min_value=0, max_value=1000, value=0)
        SupportCalls = st.number_input('Number of Support calls :', min_value=0, max_value=1000, value=0)


        input_dict={
                "Age":Age,
                "Contract Length":ContractLength,            
                "Gender":Gender,            
                "Last Interaction":LastInteraction,
                "Payment Delay":PaymentDelay,
                "Subscription Type":SubscriptionType,            
                "Support Calls":SupportCalls,
                "Tenure":Tenure,
                "Total Spend":TotalSpend,
                "Usage Frequency":UsageFrequency				
        }

        if st.button("Predict"):
            X = dv.transform([input_dict])
            y_pred = xgb_model.best_estimator_.predict(X)			
            
            churn=float(y_pred)==1.0            
            output=bool(churn)
            # st.write(f'{y_pred}')
            st.write(f'Churn: {output}')
    
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            col_list=data.columns.to_list()
            data_dict=data[col_list].to_dict(orient='records')
            X = dv.transform(data_dict)
            y_pred = xgb_model.best_estimator_.predict(X)
            churn=float(y_pred)==1.0            
            output=bool(churn)
            st.write(f'Churn: {output}')
if __name__ == '__main__':
	main()