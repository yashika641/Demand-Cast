import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from data_preprocessing import label_encoder,one_hot_encoder,ordinal_encoder,multi_label_binarizer,target_encoder,standard_scaler,robust_scaler,min_max_scaler,DROP_ROWS,fill_with_mean,ffill,bfill,fill_with_median,fill_with_mode
from train_test_split import train_test_split_data
from models import train_lstm,arima_sarima_train_test,xgboost_regression,prophet_forecast_train_test

def go_to_next_step():
    steps = list(st.session_state.steps_completed.keys())
    current_index = steps.index(st.session_state.current_step)
    if current_index < len(steps) - 1:
        st.session_state.current_step = steps[current_index + 1]

# --- Initialize session state ---
if "steps_completed" not in st.session_state:
    st.session_state.steps_completed = {
        "Upload Data": False,
        "Data Explorer": False,
        "Preprocessing": False,
        "Model Training": False,
        "Model Evaluation": False,
        "Forecast Results": False,
        "Report & Download": False
    }

if "current_step" not in st.session_state:
    st.session_state.current_step = "Upload Data"

# --- Sidebar Navigation ---
steps = list(st.session_state.steps_completed.keys())
st.sidebar.title("Workflow Navigation")
selected_step = st.sidebar.radio(
    "Choose a step:",
    steps,
    index=steps.index(st.session_state.current_step),
    format_func=lambda step: f"{'✅ ' if st.session_state.steps_completed[step] else '⬜ '} {step}"
)
#-------------------HEADER-------------------#
st.set_page_config(page_title="Demandcast", layout="wide")
st.title('DemandCast')
st.subheader('Your AI-Powered Demand Forecasting Solution')
st.markdown('Analyze ,historical sales data and predict future demand with ease.')

progress_bar = st.progress(0)
status_text = st.empty()




#-------------------UPLOAD DATA-------------------#
if selected_step=='Upload Data':
    st.subheader('Upload your data!')
    # st.warning('please upload your data in csv format ')
    uploaded_file=st.file_uploader('Upload csv',type=['csv'],key='center_uploader')
    if uploaded_file is not None:
        st.success('file uploaded successfully')
        st.session_state['uploaded_file']=uploaded_file
        st.session_state.steps_completed['Upload Data'] = True
        go_to_next_step()
    else:
        st.warning('please upload your data in csv format ')

    st.markdown('<div style="text-align: center;"> upload your csv file here<div>',unsafe_allow_html=True)

#-----------------DATA EXPLORER-------------------#
elif selected_step == 'Data Explorer':
    if not st.session_state.steps_completed['Upload Data']:
        st.warning('please upload your data first')
    else:
        uploaded_file=st.session_state['uploaded_file']
        st.subheader('Data Explorer')
        df=pd.read_csv(uploaded_file)
        st.subheader('Data Preview')
        st.dataframe(df.head(10))
        st.markdown('<div style="text-align: center;"> Data Preview<div>',unsafe_allow_html=True)
        
        st.subheader('Summary Statistics')
        st.dataframe(df.describe())
        st.markdown('<div style="text-align: center;"> Summary Statistics<div>',unsafe_allow_html=True)
        
        st.subheader('Data Types')
        st.dataframe(df.dtypes)
        st.markdown('<div style="text-align: center;"> Data Types<div>',unsafe_allow_html=True)
        
        st.subheader('Missing Values')
        st.dataframe(df.isnull().sum())
        st.markdown('<div style="text-align: center;"> Missing Values<div>',unsafe_allow_html=True)
        
        st.subheader('Data Visualization')
        numerical_cols= df.select_dtypes(include=['int64','number']).columns.to_list()
        col=st.selectbox('select the column to plot ',numerical_cols) 
        if col:
            plt.figure(figsize=(15,5))
            plt.plot(df[col])
            plt.title(f'{col}over time')
            st.pyplot(plt)
        # categorical_cols=df.select_dtypes(include=['object']).columns.to_list()

        st.markdown('<div style="text-align: center;"> Data Visualization<div>',unsafe_allow_html=True)
    
        st.session_state.steps_completed['Data Explorer'] = True
        go_to_next_step()
#--------------PREPROCESSING-------------------#
elif selected_step ==  'Preprocessing':
    if not st.session_state.steps_completed['Upload Data']:
        st.warning('please upload your data first')
    else:
        uploaded_file=st.session_state['uploaded_file']
        df= pd.read_csv(uploaded_file)
        st.info('preprocess your data to prepare it for model training')
        
        num_cols = df.select_dtypes(include=['int64','float64','number']).columns
        
        st.info('choose your numerical columns')
        final_num_cols = st.multiselect('select numerical columns',num_cols)
        
        cat_cols = df.select_dtypes(include=['object','category','bool','string']).columns
        
        st.info('choose your categorical columns')
        final_cat_cols = st.multiselect('select categorical columns',cat_cols)
        
        st.info('choose your encoder ')
        encoder= st.selectbox('select encoder',['ONE HOT ENCODER','LABEL ENCODER','MULTI LABEL ENCODER','ORDINAL ENCODER','TARGET ENCODER','NONE'])
        match encoder:
            case 'ONE HOT ENCODER':
                df_encoded=one_hot_encoder(df,final_cat_cols)
                    
            case 'LABEL ENCODER':
                df_encoded=label_encoder(df,final_cat_cols)
                    
            case 'MULTI LABEL ENCODER':
                df_encoded=multi_label_binarizer(df,final_cat_cols)
                    
            case 'ORDINAL ENCODER':
                df_encoded=ordinal_encoder(df,final_cat_cols)
                    
            case 'TARGET ENCODER':
                df_encoded=target_encoder(df,final_cat_cols)
                    
            case 'NONE':
                df_encoded=df
        
        st.info('choose your scaler')
        scaler= st.selectbox('select scaler',['MINMAX SCALER','ROBUST SCALER','STANDARD SCALER','NONE'])
        
        match scaler:
            case 'MINMAX SCALER':
                df_scaled=min_max_scaler(df_encoded,num_cols)
                    
            case 'ROBUST SCALER':
                df_scaled=robust_scaler(df_encoded,num_cols)
                    
            case 'STANDARD SCALER':
                df_scaled=standard_scaler(df_encoded,num_cols)
                    
            case 'NONE':
                df_scaled=df_encoded
        
        
        st.info('handling missing values')
        missing_options = st.selectbox('choose missing values handling method',['DROP ROWS','FILL WITH MEAN','FILL WITH MODE','FILL WITH MEDIAN','BFILL','FFILL','NONE'])
        
        match missing_options:
            case 'DROP_ROWS':
                df_final=DROP_ROWS(df_scaled)
                    
            case 'FILL WITH MEAN':
                df_final=fill_with_mean(df_scaled)
                    
            case 'FILL WITH MODE':
                df_final=fill_with_mode(df_scaled)
                    
            case 'FILL WITH MEDIAN':
                df_final=fill_with_median(df_scaled)
                    
            case 'BFILL':
                df_final=bfill(df_scaled)
                    
            case 'FFILL':
                df_final=ffill(df_scaled)
                    
            case 'NONE':
                df_final=df_scaled
                
        if 'df_final' in locals():
            if st.checkbox('drop duplicates'):
                df_final = df_final.drop_duplicates()

            if st.checkbox('show processed data'):
                st.dataframe(df_final.head(10))

            if st.button('Download processed data'):
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download processed csv",
                    data=csv,
                    file_name='processed_data.csv',
                    mime='text/csv',
                )
                st.success('CSV ready for download!')
                st.session_state.steps_completed['Preprocessing'] = True
            # Save to session state
            st.session_state['processed_df'] = df_final
            go_to_next_step()

        else:
            st.warning("No processed data found. Please select encoders/scalers and handle missing values first.")

#--------MODEL TRAINING----------#
elif selected_step == 'Model Training':
    if not st.session_state.steps_completed['Preprocessing']:
        st.warning('Please complete the Preprocessing step first.')
    
    else:
        st.subheader('model training')
        df=st.session_state['processed_df']
        st.info('choose your target column')
        col=st.selectbox('select target column',df.columns)
        st.info('Column you have chosen is: ' + str(col))
        st.info('configure model parameters and train your own model')
        st.info('choose train test split ratio')
        ratio=st.slider('train test split ratio',0.5,1.0,0.1)
        
        x_train, x_test, y_train, y_test = train_test_split_data(df, target_col=col,train_ratio=ratio)
        st.info('choose your model')
        
        model_type=st.selectbox('select model',['SARIMAX','ARIMA','LSTM','XGBRegressor','PROPHET'])
        st.subheader('SET THE PARAMETERS FOR THE MODEL')
        match model_type:
            case 'SARIMAX' | 'ARIMA' :
                p = st.slider('AR order (p)', 0, 5, 1)
                d = st.slider('Integration order (d)', 0, 2, 1)
                q = st.slider('MA order (q)', 0, 5, 1)
                P = st.slider('Seasonal AR order (P)', 0, 3, 1)
                D = st.slider('Seasonal Integration (D)', 0, 2, 1)
                Q = st.slider('Seasonal MA order (Q)', 0, 3, 1)
                m = st.slider('Seasonal period (m)', 1, 52, 7)
                seasonal=st.selectbox('seasonal', ['yes','no'])
                if st.button("Train Model"):
                    with st.spinner("Training the model..."):
                        fitted_model, forecast = arima_sarima_train_test(
                            y_test=y_test,
                            y_train=y_train,
                            seasonal=seasonal,
                            P=P, p=p,
                            Q=Q, q=q,
                            D=D, d=d,
                            m=m
                        )
                        # Save to session state
                        st.session_state['fitted_model'] = fitted_model
                        st.session_state['forecast'] = forecast

                    st.success(f"{model_type} trained successfully with selected parameters!")

                

            
            case 'LSTM':
                epochs = st.slider('Epochs', 1, 100, 10)
                batch_size = st.slider('Batch Size', 1, 128, 32)
                timesteps = st.slider('Timesteps', 1, 52, 10)
                hidden_size = st.slider('Hidden Size', 1, 100, 32)
                dropout = st.slider('Dropout', 0.0, 0.5, 0.1)
                learning_rate = st.slider('Learning Rate', 0.0001, 0.1, 0.001)
                optimizer = st.selectbox('Optimizer', ['Adam', 'SGD'])
                loss_function = st.selectbox('Loss Function', ['Mean Squared Error', 'Binary Crossentropy'])
                metrics = st.selectbox('Metrics', ['Mean Squared Error', 'Binary Crossentropy'])
                if st.button('Train Model'):
                    with st.spinner('Training the model...'):
                        model, history= train_lstm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, epochs=epochs, batch_size=batch_size, timesteps=timesteps, hidden_size=hidden_size, dropout=dropout, learning_rate=learning_rate, optimizer=optimizer, loss_function=loss_function, metrics=metrics)
                        st.session_state['model']=model
                        st.session_state['history']=history
                    st.success(f'{model_type} trained successfully with selected parameters!')
                
                
                
            case 'XGBRegressor':
                n_estimators = st.slider('Number of Estimators', 10, 1000, 100)
                max_depth = st.slider('Max Depth', 1, 10, 3)
                learning_rate = st.slider('Learning Rate', 0.01, 1.0, 0.1)
                min_child_weight = st.slider('Min Child Weight', 1, 10, 1)
                subsample = st.slider('Subsample', 0.1, 1.0, 0.5)
                colsample_bytree = st.slider('Colsample By Tree', 0.1, 1.0, 0.5)
                reg_alpha = st.slider('Alpha', 0.0, 1.0, 0.1)
                reg_lambda = st.slider('Lambda', 0.0, 1.0, 0.1)
                objective = st.selectbox('Objective', ['reg:squarederror', 'reg:squaredlogerror'])
                eval_metric = st.selectbox('Eval Metric', ['rmse', 'mae'])
                early_stopping_rounds = st.slider('Early Stopping Rounds', 1, 100, 10)
                num_boosting_rounds = st.slider('Number of Boosting Rounds', 1, 1000, 100)
                max_leaves = st.slider('Max Leaves', 1, 100, 31)
                max_bin = st.slider('Max Bin', 1, 1000, 511)
                scale_pos_weight = st.slider('Scale Pos Weight', 1, 10, 1)
                gamma = st.slider('Gamma', 0.0, 10.0, 0.0)
                lambda_l1 = st.slider('Lambda L1', 0.0, 10.0, 0.0)
                lambda_l2 = st.slider('Lambda L2', 0.0, 10.0, 0.0)
                importance_type = st.selectbox('Importance Type', ['gain', 'cover', 'total_gain', 'total_cover'])
                if st.button('Train Model'):
                    with st.spinner('Training the model...'):
                        model=xgboost_regression( x_train,y_train, x_test,y_test, n_estimators,
                                            max_depth,learning_rate,min_child_weight,
                                            subsample,colsample_bytree,reg_alpha,
                                            reg_lambda,objective,eval_metric,
                                            early_stopping_rounds,num_boosting_rounds,
                                            max_leaves,max_bin,scale_pos_weight,
                                            gamma,lambda_l1,lambda_l2,importance_type)
                        st.session_state['model']=model
                    st.success(f'{model_type} trained successfully with selected parameters!')
                
                
                
            case 'PROPHET':
                changepoint_prior_scale = st.slider('Changepoint Prior Scale', 0.001, 0.5, 0.05)
                seasonality_mode = st.selectbox('Seasonality Mode', ['additive', 'multiplicative'])
                seasonality_prior_scale = st.slider('Seasonality Prior Scale', 0.01, 10.0, 1.0)
                holidays_prior_scale = st.slider('Holidays Prior Scale', 0.01, 10.0, 1.0)
                daily_seasonality = st.selectbox('Daily Seasonality', [True, False])
                weekly_seasonality = st.selectbox('Weekly Seasonality', [True, False])
                yearly_seasonality = st.selectbox('Yearly Seasonality', [True, False])
                if st.button('Train Model'):
            
                    with st.spinner('Training the model...'):
                        model, forecast=prophet_forecast_train_test(
                                x_train, y_train,x_test,
                                y_test,changepoint_prior_scale,
                                seasonality_mode ,seasonality_prior_scale ,
                                holidays_prior_scale ,daily_seasonality ,
                                weekly_seasonality ,yearly_seasonality 
                            )
                        st.session_state['model']=model
                        st.session_state['forecast']=forecast
                        st.session_state['model_type']=model_type
                    st.success(f'{model_type} trained successfully with selected parameters!')

                st.session_state.steps_completed['Model Training'] = True
                go_to_next_step()
                
        
                
#--------------MODEL EVALUATION------------------#
elif selected_step == 'Model Evaluation':
    if not st.session_state.steps_completed['Model Training']:
        st.warning('Please complete the model training step first.')
    else:
        st.info('Evaluating the model now ...')
        model_type=st.session_state['model_type']
        reg_metrics=['MSE','RMSE','MAE','MAPE','R2','']
                
                
                
        
#--------------MODEL TRAINING------------------#
# elif selected_step == 'Model Training':
#     if not st.session_state.steps_completed['Select model Parameters']:
#         st.warning('Please complete the selecting parameters step first.')
        
#     else:
#         st.info('Training the model now ...')
#         x_train, x_test, y_train, y_test=st.session_state['x_train', 'x_test', 'y_train',' y_test']
#         model_type=st.session_state['model_type'] 
        
        
    
    

#-------------------PREDICTION-------------------#
# elif selected_step == 'Forecast Results':
#     if uploaded_file is not None:
#         st.info('choose your target column')
#         col=st.selectbox('select target column',df.columns)
#         st.info('column you have choosen is ',col)
# ------------ FORECAST RESULTS ------------------#
elif selected_step == 'Forecast Results':
    if uploaded_file:
        st.info('Forecast results will be displayed here.')
        try:
            with open('sarimax_models.pkl', 'rb') as f:
                models = pickle.load(f)
            st.success('Loaded pre-trained SARIMAX models.')
            st.write('Select series and visualize forecast.')
        except FileNotFoundError:
            st.warning('No pre-trained models found. Train a model first.')
    else:
        st.info('Please upload a CSV file to proceed.')

# -------------------- REPORT & DOWNLOAD --------------------
elif selected_step == 'Report & Download':
    st.info('Generate and download report.')
    download_button = st.button('Download Report')
    if download_button:
        st.success('Report downloaded!')
