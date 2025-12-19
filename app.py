import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, time
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="🛒 Supermarket Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS Styling
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    h1 {
        font-family: 'Playfair Display', serif !important;
        color: #e94560 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700 !important;
    }
    
    h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #edf2f4 !important;
    }
    
    .prediction-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #e94560;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.2);
        text-align: center;
        margin: 20px 0;
    }
    
    .info-card {
        background: rgba(15, 52, 96, 0.5);
        border-left: 4px solid #e94560;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 30px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# Load Model
# ============================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest model, retrain if not fitted"""
    try:
        model_path = os.path.join('ML_Model', 'random_forest_pipeline.pkl')
        pipeline = joblib.load(model_path)
        
        # Extract the Random Forest model from the pipeline
        rf_model = pipeline.named_steps.get('model', None)
        
        # Check if model is fitted
        if rf_model is None or not hasattr(rf_model, 'estimators_'):
            # Model is not fitted, need to retrain
            st.warning("⚠️ Model not fitted. Retraining model from training data...")
            return retrain_model()
        
        return rf_model, pipeline
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {str(e)}. Attempting to retrain...")
        return retrain_model()

@st.cache_resource
def retrain_model():
    """Retrain the model using the training data"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        # Load training data
        data_path = os.path.join('Data_Sets', 'SuperMarket Analysis.csv')
        if not os.path.exists(data_path):
            st.error("❌ Training data not found! Please ensure 'Data_Sets/SuperMarket Analysis.csv' exists.")
            return None, None
        
        df = pd.read_csv(data_path)
        
        # Preprocess data (same as notebook)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
        
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Hour'] = df['Time'].apply(lambda x: x.hour)
        df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)
        
        df['ProductLine_TimeOfDay'] = df['Product line'].astype(str) + '_' + df['TimeOfDay'].astype(str)
        df['ProductLine_Gender'] = df['Product line'].astype(str) + '_' + df['Gender'].astype(str)
        df['Branch_TimeOfDay'] = df['Branch'].astype(str) + '_' + df['TimeOfDay'].astype(str)
        
        df = df.drop(columns=['Date', 'Time', 'Invoice ID'], errors='ignore')
        df = df.drop(columns=['Tax 5%', 'cogs', 'gross margin percentage', 'gross income'], errors='ignore')
        
        # Separate features and target
        X = df.drop(columns=['Sales'])
        y = df['Sales']
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(exclude=['number', 'int', 'float']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number', 'int', 'float']).columns.tolist()
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", StandardScaler(), numerical_cols)
            ]
        )
        
        # Create and fit pipeline
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", rf_model)
        ])
        
        # Fit the pipeline
        pipeline.fit(X, y)
        
        # Save the retrained model
        model_path = os.path.join('ML_Model', 'random_forest_pipeline.pkl')
        os.makedirs('ML_Model', exist_ok=True)
        joblib.dump(pipeline, model_path)
        
        st.success("✅ Model retrained and saved successfully!")
        
        return rf_model, pipeline
        
    except Exception as e:
        st.error(f"❌ Error retraining model: {str(e)}")
        import traceback
        st.exception(e)
        return None, None

# ============================================
# Preprocessing Functions
# ============================================
def get_time_of_day(hour):
    """Determine the time of day based on hour"""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def preprocess_data(df):
    """
    Preprocess data to match the exact format used during training.
    This replicates the preprocessing from the Jupyter notebook.
    """
    df = df.copy()
    
    # Convert Date and Time if they exist as strings
    if 'Date' in df.columns:
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])
        elif not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
    
    if 'Time' in df.columns:
        if df['Time'].dtype == 'object':
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p', errors='coerce').dt.time
            except:
                try:
                    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time
                except:
                    df['Time'] = time(12, 0)
    
    # Extract date/time features
    if 'Date' in df.columns:
        df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['Day'] = pd.to_datetime(df['Date']).dt.day
        df['Month'] = pd.to_datetime(df['Date']).dt.month
    else:
        df['DayOfWeek'] = 0
        df['Day'] = 1
        df['Month'] = 1
    
    if 'Time' in df.columns:
        df['Hour'] = df['Time'].apply(
            lambda x: x.hour if isinstance(x, time) 
            else pd.to_datetime(x, format='%H:%M:%S').hour if isinstance(x, str) 
            else 12
        )
        df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)
    else:
        df['Hour'] = 12
        df['TimeOfDay'] = 'Afternoon'
    
    # Create interaction features (matching notebook cell 51)
    df['ProductLine_TimeOfDay'] = df['Product line'].astype(str) + '_' + df['TimeOfDay'].astype(str)
    df['ProductLine_Gender'] = df['Product line'].astype(str) + '_' + df['Gender'].astype(str)
    df['Branch_TimeOfDay'] = df['Branch'].astype(str) + '_' + df['TimeOfDay'].astype(str)
    
    # Drop Date and Time columns (matching notebook cell 22)
    df = df.drop(columns=['Date', 'Time'], errors='ignore')
    
    # Drop any columns that shouldn't be in the model (like Current Sales if present)
    df = df.drop(columns=['Current Sales'], errors='ignore')
    
    return df

def prepare_input_for_model(df):
    """
    Prepare input DataFrame to match the exact column structure expected by the model.
    This ensures all one-hot encoded columns exist and numerical columns are scaled.
    """
    # Define all possible categorical values (from the dataset)
    branch_values = ['Alex', 'Cairo', 'Giza']
    city_values = ['Mandalay', 'Naypyitaw', 'Yangon']
    customer_type_values = ['Member', 'Normal']
    gender_values = ['Female', 'Male']
    product_line_values = ['Electronic accessories', 'Fashion accessories', 
                          'Food and beverages', 'Health and beauty', 
                          'Home and lifestyle', 'Sports and travel']
    payment_values = ['Cash', 'Credit card', 'Ewallet']
    time_of_day_values = ['Afternoon', 'Evening', 'Morning', 'Night']
    
    # Generate all possible interaction feature values
    product_time_values = sorted([f"{p}_{t}" for p in product_line_values for t in time_of_day_values])
    product_gender_values = sorted([f"{p}_{g}" for p in product_line_values for g in gender_values])
    branch_time_values = sorted([f"{b}_{t}" for b in branch_values for t in time_of_day_values])
    
    # Numerical columns (these will be scaled)
    numerical_cols = ['Unit price', 'Quantity', 'Rating', 'DayOfWeek', 'Day', 'Month', 'Hour']
    
    # Categorical columns for one-hot encoding
    categorical_cols = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 
                        'Payment', 'TimeOfDay', 'ProductLine_TimeOfDay', 
                        'ProductLine_Gender', 'Branch_TimeOfDay']
    
    # One-hot encode categorical variables (matching notebook cell 59)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Build the full column list (all possible one-hot encoded columns)
    all_columns = numerical_cols.copy()
    
    # Add one-hot encoded columns in the order they would appear
    for val in branch_values:
        all_columns.append(f'Branch_{val}')
    for val in city_values:
        all_columns.append(f'City_{val}')
    for val in customer_type_values:
        all_columns.append(f'Customer type_{val}')
    for val in gender_values:
        all_columns.append(f'Gender_{val}')
    for val in product_line_values:
        all_columns.append(f'Product line_{val}')
    for val in payment_values:
        all_columns.append(f'Payment_{val}')
    for val in time_of_day_values:
        all_columns.append(f'TimeOfDay_{val}')
    all_columns.extend([f'ProductLine_TimeOfDay_{val}' for val in product_time_values])
    all_columns.extend([f'ProductLine_Gender_{val}' for val in product_gender_values])
    all_columns.extend([f'Branch_TimeOfDay_{val}' for val in branch_time_values])
    
    # Create a DataFrame with all columns, initialized to 0
    result_df = pd.DataFrame(0, index=df_encoded.index, columns=all_columns)
    
    # Fill in the values from the encoded dataframe
    for col in df_encoded.columns:
        if col in result_df.columns:
            result_df[col] = df_encoded[col].values
    
    # Scale numerical columns (using StandardScaler as in notebook)
    # Note: Ideally we'd use the scaler fitted on training data, but since pipeline wasn't fitted,
    # we'll fit on current data. For better results, you should retrain the pipeline.
    scaler = StandardScaler()
    result_df[numerical_cols] = scaler.fit_transform(result_df[numerical_cols])
    
    return result_df

# ============================================
# Main Application
# ============================================
def main():
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1>🛒 Supermarket Sales Predictor</h1>
            <p style="font-size: 1.2em; color: #8d99ae;">
                Predict sales for multiple transactions using AI-powered machine learning
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    rf_model, pipeline = load_model()
    if rf_model is None:
        st.stop()
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📊 Tabular Input", "📝 Single Entry"])
    
    with tab1:
        st.markdown("### Upload CSV or Enter Data Manually")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload CSV file with transaction data",
            type=['csv'],
            help="CSV should contain columns: Branch, City, Customer type, Gender, Product line, Unit price, Quantity, Payment, Rating, Date, Time"
        )
        
        # Manual data entry option
        st.markdown("---")
        st.markdown("#### Or Enter Data Manually")
        
        num_rows = st.number_input("Number of transactions", min_value=1, max_value=50, value=3)
        
        # Create input form
        input_data = []
        
        for i in range(num_rows):
            with st.expander(f"Transaction {i+1}", expanded=(i == 0)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    branch = st.selectbox(f"Branch {i+1}", ["Alex", "Cairo", "Giza"], key=f"branch_{i}")
                    city = st.selectbox(f"City {i+1}", ["Yangon", "Mandalay", "Naypyitaw"], key=f"city_{i}")
                    customer_type = st.selectbox(f"Customer Type {i+1}", ["Member", "Normal"], key=f"cust_{i}")
                    gender = st.selectbox(f"Gender {i+1}", ["Female", "Male"], key=f"gender_{i}")
                
                with col2:
                    product_line = st.selectbox(
                        f"Product Line {i+1}",
                        ["Health and beauty", "Electronic accessories", "Home and lifestyle",
                         "Sports and travel", "Food and beverages", "Fashion accessories"],
                        key=f"product_{i}"
                    )
                    unit_price = st.number_input(f"Unit Price {i+1}", min_value=10.0, max_value=100.0, value=50.0, step=0.01, key=f"price_{i}")
                    quantity = st.number_input(f"Quantity {i+1}", min_value=1, max_value=10, value=5, step=1, key=f"qty_{i}")
                
                with col3:
                    payment = st.selectbox(f"Payment {i+1}", ["Ewallet", "Cash", "Credit card"], key=f"pay_{i}")
                    rating = st.slider(f"Rating {i+1}", min_value=1.0, max_value=10.0, value=7.0, step=0.1, key=f"rating_{i}")
                    transaction_date = st.date_input(f"Date {i+1}", value=datetime.now(), key=f"date_{i}")
                    transaction_time = st.time_input(f"Time {i+1}", value=time(12, 0), key=f"time_{i}")
                
                # Optional: Current Sales (if known)
                current_sales = st.number_input(
                    f"Current Sales (Optional) {i+1}",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    key=f"current_sales_{i}",
                    help="Enter actual sales if known, leave 0 for prediction only"
                )
                
                input_data.append({
                    'Branch': branch,
                    'City': city,
                    'Customer type': customer_type,
                    'Gender': gender,
                    'Product line': product_line,
                    'Unit price': unit_price,
                    'Quantity': quantity,
                    'Payment': payment,
                    'Rating': rating,
                    'Date': transaction_date,
                    'Time': transaction_time,
                    'Current Sales': current_sales
                })
        
        # Process button
        if st.button("🔮 Predict Sales for All Transactions", use_container_width=True):
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.DataFrame(input_data)
            
            if df.empty:
                st.warning("⚠️ No data to process!")
            else:
                with st.spinner('🔄 Processing transactions and making predictions...'):
                    try:
                        # Create a copy for processing
                        process_df = df.copy()
                        
                        # Preprocess the data
                        processed_df = preprocess_data(process_df)
                        
                        # Use the pipeline to predict (it handles preprocessing internally)
                        predictions = pipeline.predict(processed_df)
                        
                        # Create results DataFrame
                        results_df = df.copy()
                        results_df['Predicted Sales'] = predictions
                        
                        # Calculate additional metrics
                        results_df['Base Amount'] = results_df['Unit price'] * results_df['Quantity']
                        results_df['Predicted Tax (5%)'] = results_df['Predicted Sales'] * 0.05 / 1.05
                        results_df['Predicted COGS'] = results_df['Predicted Sales'] / 1.05
                        
                        # If current sales provided, calculate difference
                        if 'Current Sales' in results_df.columns:
                            results_df['Difference'] = results_df['Predicted Sales'] - results_df['Current Sales']
                            results_df['Error %'] = ((results_df['Difference'] / results_df['Current Sales']) * 100).round(2)
                            results_df.loc[results_df['Current Sales'] == 0, 'Error %'] = np.nan
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Prediction Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Transactions", len(results_df))
                        with col2:
                            st.metric("Total Predicted Sales", f"${results_df['Predicted Sales'].sum():,.2f}")
                        with col3:
                            if 'Current Sales' in results_df.columns and results_df['Current Sales'].sum() > 0:
                                st.metric("Total Current Sales", f"${results_df['Current Sales'].sum():,.2f}")
                        with col4:
                            if 'Difference' in results_df.columns:
                                avg_error = results_df['Error %'].mean()
                                if not np.isnan(avg_error):
                                    st.metric("Avg Error %", f"{avg_error:.2f}%")
                        
                        # Display table
                        st.markdown("### Detailed Results")
                        
                        # Select columns to display
                        display_cols = ['Branch', 'City', 'Product line', 'Unit price', 'Quantity', 
                                       'Payment', 'Rating', 'Predicted Sales']
                        
                        if 'Current Sales' in results_df.columns:
                            display_cols.insert(-1, 'Current Sales')
                            if 'Difference' in results_df.columns:
                                display_cols.extend(['Difference', 'Error %'])
                        
                        display_cols.extend(['Base Amount', 'Predicted Tax (5%)', 'Predicted COGS'])
                        
                        # Format the display DataFrame
                        display_df = results_df[display_cols].copy()
                        display_df = display_df.round(2)
                        
                        st.dataframe(display_df, use_container_width=True, height=400)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"sales_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("✅ Predictions completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
                        st.exception(e)
    
    with tab2:
        st.markdown("### Single Transaction Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            branch = st.selectbox("Branch", ["Alex", "Cairo", "Giza"])
            city = st.selectbox("City", ["Yangon", "Mandalay", "Naypyitaw"])
            customer_type = st.selectbox("Customer Type", ["Member", "Normal"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            product_line = st.selectbox(
                "Product Line",
                ["Health and beauty", "Electronic accessories", "Home and lifestyle",
                 "Sports and travel", "Food and beverages", "Fashion accessories"]
            )
        
        with col2:
            unit_price = st.number_input("Unit Price ($)", min_value=10.0, max_value=100.0, value=50.0, step=0.01)
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=5, step=1)
            payment = st.selectbox("Payment Method", ["Ewallet", "Cash", "Credit card"])
            rating = st.slider("Customer Rating", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
            transaction_date = st.date_input("Transaction Date", value=datetime.now())
            transaction_time = st.time_input("Transaction Time", value=time(12, 0))
        
        current_sales = st.number_input(
            "Current Sales (Optional)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            help="Enter actual sales if known, leave 0 for prediction only"
        )
        
        if st.button("🔮 Predict Sales", use_container_width=True):
            with st.spinner('🔄 Analyzing transaction data...'):
                try:
                    input_df = pd.DataFrame([{
                        'Branch': branch,
                        'City': city,
                        'Customer type': customer_type,
                        'Gender': gender,
                        'Product line': product_line,
                        'Unit price': unit_price,
                        'Quantity': quantity,
                        'Payment': payment,
                        'Rating': rating,
                        'Date': transaction_date,
                        'Time': transaction_time
                    }])
                    
                    # Preprocess
                    processed_df = preprocess_data(input_df)
                    
                    # Use the pipeline to predict (it handles preprocessing internally)
                    prediction = pipeline.predict(processed_df)[0]
                    
                    # Display result
                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Result")
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h3 style="color: #8d99ae;">Predicted Sales</h3>
                                <h1 style="font-size: 4rem; color: #e94560;">
                                    ${prediction:,.2f}
                                </h1>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        if current_sales > 0:
                            st.markdown(f"""
                                <div class="prediction-card">
                                    <h3 style="color: #8d99ae;">Current Sales</h3>
                                    <h1 style="font-size: 4rem; color: #4CAF50;">
                                        ${current_sales:,.2f}
                                    </h1>
                                    <p style="color: #8d99ae;">
                                        Difference: ${prediction - current_sales:,.2f}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.metric("Base Amount", f"${unit_price * quantity:.2f}")
                        st.metric("Estimated Tax (5%)", f"${prediction * 0.05 / 1.05:.2f}")
                        st.metric("Estimated COGS", f"${prediction / 1.05:.2f}")
                    
                    st.success("✅ Prediction completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.exception(e)
    
    # Information section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
            ### How It Works
            
            This application uses a **Random Forest Regression** model trained on supermarket sales data.
            
            #### Features:
            - **Tabular Input**: Upload CSV or enter multiple transactions manually
            - **Single Entry**: Quick prediction for one transaction
            - **Comparison**: View both current and predicted sales side by side
            
            #### Model Performance:
            - **R² Score:** ~99.75% (Cross-validated)
            - **Algorithm:** Random Forest Regressor with 100 estimators
        """)

if __name__ == "__main__":
    main()
