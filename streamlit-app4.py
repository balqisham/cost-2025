import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import linregress
from sklearn.impute import KNNImputer

def main():
    st.title('💲Cost Prediction RT2025💲')
    
    # File upload
    st.sidebar.header('Upload Data')
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Reset predictions list when a new file is uploaded
        if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
            st.session_state['predictions'] = []
            st.session_state['last_uploaded_file'] = uploaded_file.name
        
        # Read and display the data
        df = pd.read_csv(uploaded_file)
        
        # Handle missing data using KNN Imputer
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # Data Overview Section
        st.header('Data Overview')
        st.write('Dataset Shape:', df_imputed.shape)
        st.dataframe(df_imputed.head())
        
        # Prepare the data
        X = df_imputed.iloc[:, :-1]
        y = df_imputed.iloc[:, -1]
        target_column = y.name
        
        # Model Training Section
        st.header('Model Training')
        
        # Allow user to adjust test size
        test_size = st.slider('Select test size (0.0-1.0)', 0.1, 0.5, 0.2)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Model Performance Section
        st.header('Model Performance')
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('RMSE', f'{rmse:.2f}')
        with col2:
            st.metric('R² Score', f'{r2:.2f}')
        
        # Visualization Section
        with st.expander('Data Visualization', expanded=True):
            st.subheader('Correlation Matrix')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(fig)
            plt.close()
            
            st.subheader('Feature Importance')
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title('Feature Importance')
            st.pyplot(fig)
            plt.close()
            
            st.subheader('Cost Curve (Feature vs Target Scatter Plot)')
            feature_to_plot = st.selectbox('Cost Curve Dropdown Menu', X.columns)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_imputed, x=feature_to_plot, y=target_column)
            
            # Calculate the best fit linear line
            slope, intercept, r_value, p_value, std_err = linregress(df_imputed[feature_to_plot], y)
            plt.plot(df_imputed[feature_to_plot], intercept + slope * df_imputed[feature_to_plot], color='red', label='Best Fit Line')
            
            # Add linear equation text to the plot
            equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
            plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            
            plt.xlabel(feature_to_plot)
            plt.ylabel(target_column)
            plt.legend()
            
            st.pyplot(fig)
        
        # Prediction Section
        st.header('Make New Predictions')
        st.write('Enter values for prediction:')
        
        # Input field for project name
        project_name = st.text_input('Enter Project Name')
        
        # Create input fields for each feature
        new_data = {}
        for column in X.columns:
            new_data[column] = st.number_input(f'Enter {column}', 
                                             value=float(X[column].mean()),
                                             step=float(X[column].std()))
        
        # Initialize or load predictions list
        if 'predictions' not in st.session_state:
            st.session_state['predictions'] = []
        
        if st.button('Predict'):
            # Prepare the input data
            input_data = pd.DataFrame([new_data])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = rf_model.predict(input_scaled)
            
            # Store the prediction
            prediction_entry = {'Project Name': project_name}
            prediction_entry.update(new_data)
            prediction_entry[target_column] = round(prediction[0], 2)
            st.session_state['predictions'].append(prediction_entry)
            
            st.success(f'{project_name}\n{target_column}: {prediction[0]:,.2f}')
        
        # Simplified Project List Section
        with st.expander('Simplified Project List', expanded=True):
            if st.session_state['predictions']:
                for i, pred in enumerate(st.session_state['predictions']):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{i + 1}. {pred['Project Name']}")
                    with col2:
                        if st.button(f"Delete", key=f"delete_{i}"):
                            st.session_state['predictions'].pop(i)
                            st.rerun()
        
        # Updated Cost Curve with New Prediction
        with st.expander('Updated Cost Curve with New Prediction', expanded=True):
            updated_feature_to_plot = st.selectbox('Updated Cost Curve Dropdown Menu', X.columns, key='updated_feature')
            
            if updated_feature_to_plot:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=df_imputed, x=updated_feature_to_plot, y=target_column)
                
                # Recalculate the best fit linear line for the selected feature
                slope, intercept, r_value, p_value, std_err = linregress(df_imputed[updated_feature_to_plot], y)
                plt.plot(df_imputed[updated_feature_to_plot], intercept + slope * df_imputed[updated_feature_to_plot], color='red', label='Best Fit Line')
                
                # Add linear equation text to the plot
                equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
                plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
                         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                
                if 'predictions' in st.session_state and st.session_state['predictions']:
                    for pred in st.session_state['predictions']:
                        plt.scatter(pred[updated_feature_to_plot], pred[target_column], color='red', marker='x', s=100)
                        plt.annotate(pred['Project Name'], (pred[updated_feature_to_plot], pred[target_column]), textcoords="offset points", xytext=(5,5), ha='center')
                
                plt.xlabel(updated_feature_to_plot)
                plt.ylabel(target_column)
                plt.legend()
                st.pyplot(fig)
        
        # Display all predictions in a table and provide an option to download as Excel file
        st.header('Project Prediction List')
        
        if st.session_state['predictions']:
            predictions_df = pd.DataFrame(st.session_state['predictions'])
            predictions_df = predictions_df.applymap(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x)
            st.table(predictions_df)

            # Provide an option to download the predictions as an Excel file
            if st.button('Download Predictions as Excel'):
                excel_file_name = uploaded_file.name.replace('.csv', '_predictions.xlsx')
                predictions_df.to_excel(excel_file_name, index=False)
                with open(excel_file_name, "rb") as file:
                    btn = st.download_button(
                        label="Download Excel",
                        data=file,
                        file_name=excel_file_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

if __name__ == '__main__':
    main()
