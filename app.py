import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Initialize session state for storing entries
    if 'entries' not in st.session_state:
        st.session_state.entries = []
    
    # Month selection for prediction target (moved before CSV upload)
    st.markdown("### 🗓️ Prediction Target Month")
    col_month1, col_month2 = st.columns(2)
    with col_month1:
        prediction_year = st.number_input("Target Year", min_value=2020, max_value=2030, value=datetime.now().year, key="pred_year")
    with col_month2:
        prediction_month = st.selectbox(
            "Target Month",
            options=[
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
            index=(datetime.now().month) % 12,  # Default to next month
            key="pred_month"
        )
    
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    target_month_num = month_map[prediction_month]
    default_month = f"{prediction_month} {prediction_year}"
    
    st.info(f"📅 **Predicting sales for: {prediction_month} {prediction_year}**")
    
    st.markdown("---")
    
    # File upload option
    st.markdown("### 📤 Upload CSV (Optional)")
    
    # CSV Format Information
    with st.expander("📋 CSV/Excel File Format Requirements", expanded=False):
        st.markdown("""
        **Required Columns (in order):**
        
        | Column Name | Data Type | Example Values | Required |
        |------------|-----------|----------------|----------|
        | Branch | Text | Alex, Cairo, Giza | ✅ Yes |
        | City | Text | Yangon, Mandalay, Naypyitaw | ✅ Yes |
        | Customer type | Text | Member, Normal | ✅ Yes |
        | Gender | Text | Female, Male | ✅ Yes |
        | Product line | Text | Health and beauty, Electronic accessories, etc. | ✅ Yes |
        | Unit price | Number | 10.0 - 100.0 | ✅ Yes |
        | Quantity | Integer | 1 - 10 | ✅ Yes |
        | Payment | Text | Ewallet, Cash, Credit card | ✅ Yes |
        | Rating | Number | 1.0 - 10.0 | ✅ Yes |
        | Date | Date | 1/5/2019, 2019-01-05 | ✅ Yes |
        | Time | Time | 1:08:00 PM, 13:08:00 | ✅ Yes |
        | Target Month | Text (Optional) | January 2024 | ❌ No |
        | Current Sales | Number (Optional) | 548.97 | ❌ No |
        
        **Notes:**
        - If `Current Sales` is not provided, it will be calculated as: `Unit price × Quantity × 1.05`
        - If `Target Month` is not provided, entries will use the selected month from the form above
        - Date format: MM/DD/YYYY or YYYY-MM-DD
        - Time format: HH:MM:SS AM/PM or HH:MM:SS (24-hour)
        - All text values are case-sensitive (e.g., "Member" not "member")
        """)
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with transaction data",
        type=['csv'],
        help="CSV should contain columns: Branch, City, Customer type, Gender, Product line, Unit price, Quantity, Payment, Rating, Date, Time"
    )
    
    if uploaded_file:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            # Calculate current sales for uploaded data
            if 'Current Sales' not in df_uploaded.columns:
                df_uploaded['Current Sales'] = df_uploaded['Unit price'] * df_uploaded['Quantity'] * 1.05
            else:
                # Ensure Current Sales is numeric
                df_uploaded['Current Sales'] = pd.to_numeric(df_uploaded['Current Sales'], errors='coerce')
                df_uploaded['Current Sales'] = df_uploaded['Current Sales'].fillna(
                    df_uploaded['Unit price'] * df_uploaded['Quantity'] * 1.05
                )
            
            # Convert to list of dicts and add to entries
            for _, row in df_uploaded.iterrows():
                # Handle Date column
                date_val = row.get('Date', datetime.now())
                if isinstance(date_val, str):
                    try:
                        date_val = pd.to_datetime(date_val).date()
                    except:
                        date_val = datetime.now().date()
                elif pd.isna(date_val):
                    date_val = datetime.now().date()
                else:
                    try:
                        date_val = pd.to_datetime(date_val).date()
                    except:
                        date_val = datetime.now().date()
                
                # Handle Time column
                time_val = row.get('Time', '12:00:00 PM')
                if isinstance(time_val, str):
                    try:
                        time_val = pd.to_datetime(time_val, format='%I:%M:%S %p').time()
                    except:
                        try:
                            time_val = pd.to_datetime(time_val, format='%H:%M:%S').time()
                        except:
                            time_val = time(12, 0)
                elif pd.isna(time_val):
                    time_val = time(12, 0)
                else:
                    time_val = time(12, 0)
                
                # Get Target Month from CSV if available, otherwise use form selection
                target_month_val = str(row.get('Target Month', default_month))
                if target_month_val == 'nan' or not target_month_val or target_month_val.strip() == '':
                    target_month_val = default_month
                
                entry = {
                    'Branch': str(row.get('Branch', 'Alex')),
                    'City': str(row.get('City', 'Yangon')),
                    'Customer type': str(row.get('Customer type', 'Member')),
                    'Gender': str(row.get('Gender', 'Female')),
                    'Product line': str(row.get('Product line', 'Health and beauty')),
                    'Unit price': float(row.get('Unit price', 50.0)),
                    'Quantity': int(row.get('Quantity', 1)),
                    'Payment': str(row.get('Payment', 'Ewallet')),
                    'Rating': float(row.get('Rating', 7.0)),
                    'Date': date_val,
                    'Time': time_val,
                    'Current Sales': float(row.get('Current Sales', row.get('Unit price', 50.0) * row.get('Quantity', 1) * 1.05)),
                    'Target Month': target_month_val
                }
                st.session_state.entries.append(entry)
            st.success(f"✅ Loaded {len(df_uploaded)} entries from CSV!")
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
            import traceback
            st.exception(e)
    
    st.markdown("---")
    st.markdown("### 📝 Add Transaction Entry")
    
    st.markdown("---")
    st.markdown("#### 📋 Transaction Details")
    
    # Single form for data entry
    col1, col2 = st.columns(2)
    
    with col1:
        branch = st.selectbox("Branch", ["Alex", "Cairo", "Giza"], key="form_branch")
        city = st.selectbox("City", ["Yangon", "Mandalay", "Naypyitaw"], key="form_city")
        customer_type = st.selectbox("Customer Type", ["Member", "Normal"], key="form_customer")
        gender = st.selectbox("Gender", ["Female", "Male"], key="form_gender")
        product_line = st.selectbox(
            "Product Line",
            ["Health and beauty", "Electronic accessories", "Home and lifestyle",
             "Sports and travel", "Food and beverages", "Fashion accessories"],
            key="form_product"
        )
    
    with col2:
        unit_price = st.number_input("Unit Price ($)", min_value=10.0, max_value=100.0, value=50.0, step=0.01, key="form_price")
        quantity = st.number_input("Quantity", min_value=1, max_value=10, value=5, step=1, key="form_qty")
        payment = st.selectbox("Payment Method", ["Ewallet", "Cash", "Credit card"], key="form_payment")
        rating = st.slider("Customer Rating", min_value=1.0, max_value=10.0, value=7.0, step=0.1, key="form_rating")
        # Use target month for transaction date
        transaction_date = st.date_input(
            "Transaction Date (in target month)", 
            value=datetime(prediction_year, target_month_num, 15), 
            key="form_date"
        )
        transaction_time = st.time_input("Transaction Time", value=time(12, 0), key="form_time")
    
    # Calculate current sales automatically
    base_amount = unit_price * quantity
    current_sales = base_amount * 1.05  # Adding 5% tax
    
    # Display calculated current sales
    st.info(f"💰 **Calculated Current Sales:** ${current_sales:,.2f} (Base: ${base_amount:,.2f} + 5% Tax)")
    
    # Add Entry button
    col_add1, col_add2, col_add3 = st.columns([1, 2, 1])
    with col_add2:
        if st.button("➕ Add Entry", use_container_width=True, type="primary"):
            entry = {
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
                'Current Sales': current_sales,
                'Target Month': f"{prediction_month} {prediction_year}"
            }
            st.session_state.entries.append(entry)
            st.success(f"✅ Entry added! Total entries: {len(st.session_state.entries)}")
            st.rerun()
    
    st.markdown("---")
    
    # Display entries table
    if len(st.session_state.entries) > 0:
        st.markdown("### 📊 Transaction Entries")
        
        # Create DataFrame from entries
        entries_df = pd.DataFrame(st.session_state.entries)
        
        # Display the table
        st.dataframe(entries_df, use_container_width=True, height=300)
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            if st.button("🗑️ Clear All Entries", use_container_width=True):
                st.session_state.entries = []
                st.rerun()
        
        with col_btn2:
            if st.button("🔮 Predict Sales", use_container_width=True, type="primary"):
                with st.spinner('🔄 Processing transactions and making predictions...'):
                    try:
                        # Create DataFrame from entries
                        df = pd.DataFrame(st.session_state.entries)
                        
                        # Preprocess the data
                        processed_df = preprocess_data(df)
                        
                        # Use the pipeline to predict (it handles preprocessing internally)
                        predictions = pipeline.predict(processed_df)
                        
                        # Create results DataFrame
                        results_df = df.copy()
                        results_df['Predicted Sales'] = predictions
                        
                        # Calculate additional metrics
                        results_df['Base Amount'] = results_df['Unit price'] * results_df['Quantity']
                        results_df['Current Tax (5%)'] = results_df['Current Sales'] * 0.05 / 1.05
                        results_df['Predicted Tax (5%)'] = results_df['Predicted Sales'] * 0.05 / 1.05
                        results_df['Predicted COGS'] = results_df['Predicted Sales'] / 1.05
                        
                        # Calculate difference and error
                        results_df['Difference'] = results_df['Predicted Sales'] - results_df['Current Sales']
                        results_df['Error %'] = ((results_df['Difference'] / results_df['Current Sales']) * 100).round(2)
                        
                        # Display results
                        st.markdown("---")
                        
                        # Get target month from entries
                        target_month = results_df['Target Month'].iloc[0] if 'Target Month' in results_df.columns else "Next Month"
                        
                        st.markdown(f"## 📊 Sales Prediction for {target_month}")
                        st.markdown(f"### Forecasted sales based on {len(results_df)} transaction(s)")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Transactions", len(results_df))
                        with col2:
                            st.metric("Current Month Sales", f"${results_df['Current Sales'].sum():,.2f}")
                        with col3:
                            st.metric(f"Predicted Sales ({target_month})", f"${results_df['Predicted Sales'].sum():,.2f}")
                        with col4:
                            avg_error = abs(results_df['Error %']).mean()
                            st.metric("Avg Error %", f"{avg_error:.2f}%")
                        
                        # Monthly projection
                        st.markdown("---")
                        st.markdown("### 📈 Monthly Sales Projection")
                        
                        # Calculate monthly totals
                        monthly_current = results_df['Current Sales'].sum()
                        monthly_predicted = results_df['Predicted Sales'].sum()
                        monthly_change = monthly_predicted - monthly_current
                        monthly_change_pct = ((monthly_change / monthly_current) * 100) if monthly_current > 0 else 0
                        
                        proj_col1, proj_col2 = st.columns(2)
                        with proj_col1:
                            st.markdown(f"""
                                <div class="info-card">
                                    <h4 style="color: #e94560; margin: 0;">Current Month Total</h4>
                                    <h2 style="color: #4CAF50; margin: 10px 0;">${monthly_current:,.2f}</h2>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with proj_col2:
                            st.markdown(f"""
                                <div class="info-card">
                                    <h4 style="color: #e94560; margin: 0;">{target_month} Projection</h4>
                                    <h2 style="color: #e94560; margin: 10px 0;">${monthly_predicted:,.2f}</h2>
                                    <p style="color: #8d99ae; margin: 5px 0;">
                                        Change: ${monthly_change:,.2f} ({monthly_change_pct:+.2f}%)
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Calculate total sales for comparison
                        total_current = results_df['Current Sales'].sum()
                        total_predicted = results_df['Predicted Sales'].sum()
                        total_difference = total_predicted - total_current
                        total_change_percent = ((total_difference / total_current) * 100) if total_current > 0 else 0
                        
                        # Display message based on prediction for next month
                        st.markdown("---")
                        if total_predicted < total_current:
                            st.warning(f"⚠️ **Sales Decrease Predicted for {target_month}**: The model predicts a decrease of ${abs(total_difference):,.2f} ({abs(total_change_percent):.2f}%) compared to current month sales. Predicted sales for {target_month}: ${total_predicted:,.2f} vs Current month: ${total_current:,.2f}")
                        elif total_predicted > total_current:
                            st.success(f"📈 **Sales Increase Predicted for {target_month}**: The model predicts an increase of ${total_difference:,.2f} ({total_change_percent:.2f}%) compared to current month sales. Predicted sales for {target_month}: ${total_predicted:,.2f} vs Current month: ${total_current:,.2f}")
                        else:
                            st.info(f"➡️ **Sales Stable for {target_month}**: The model predicts sales will remain similar. Predicted sales for {target_month}: ${total_predicted:,.2f} vs Current month: ${total_current:,.2f}")
                        
                        # Group by Target Month for separate comparisons
                        # Ensure Target Month column exists and is string type
                        if 'Target Month' not in results_df.columns:
                            results_df['Target Month'] = target_month
                        
                        # Convert Target Month to string and handle any NaN values
                        results_df['Target Month'] = results_df['Target Month'].astype(str)
                        results_df['Target Month'] = results_df['Target Month'].replace('nan', target_month)
                        results_df['Target Month'] = results_df['Target Month'].fillna(target_month)
                        
                        unique_months = results_df['Target Month'].unique()
                        
                        # Create separate sections for each target month
                        for month in unique_months:
                            # Ensure month is a string
                            month_str = str(month) if not isinstance(month, str) else month
                            month_df = results_df[results_df['Target Month'] == month_str].copy()
                            
                            if len(month_df) == 0:
                                continue
                            
                            st.markdown(f"### 📊 Sales Comparison Chart - {month_str}")
                            
                            # Prepare data for bar chart - Group by Product Line
                            # Ensure Product line is string type
                            month_df['Product line'] = month_df['Product line'].astype(str)
                            
                            product_line_data = month_df.groupby('Product line').agg({
                                'Current Sales': 'sum',
                                'Predicted Sales': 'sum'
                            }).reset_index()
                            
                            # Ensure numeric columns are proper types
                            product_line_data['Current Sales'] = pd.to_numeric(product_line_data['Current Sales'], errors='coerce').fillna(0)
                            product_line_data['Predicted Sales'] = pd.to_numeric(product_line_data['Predicted Sales'], errors='coerce').fillna(0)
                            
                            # Calculate monthly totals for this month
                            month_current = month_df['Current Sales'].sum()
                            month_predicted = month_df['Predicted Sales'].sum()
                            month_diff = month_predicted - month_current
                            month_change_pct = ((month_diff / month_current) * 100) if month_current > 0 else 0
                            
                            # Create bar chart
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                            
                            # Chart 1: Product Line comparison
                            x = np.arange(len(product_line_data))
                            width = 0.35
                            
                            bars1 = ax1.bar(x - width/2, product_line_data['Current Sales'], width, 
                                          label='Current Month Sales', color='#4CAF50', alpha=0.8)
                            bars2 = ax1.bar(x + width/2, product_line_data['Predicted Sales'], width, 
                                          label=f'{month_str} Predicted', color='#e94560', alpha=0.8)
                            
                            ax1.set_xlabel('Product Line', fontsize=12, fontweight='bold')
                            ax1.set_ylabel('Sales Amount ($)', fontsize=12, fontweight='bold')
                            ax1.set_title(f'Sales by Product Line - {month_str}', fontsize=13, fontweight='bold', pad=15)
                            ax1.set_xticks(x)
                            ax1.set_xticklabels(product_line_data['Product line'], rotation=45, ha='right')
                            ax1.legend(fontsize=10)
                            ax1.grid(axis='y', alpha=0.3, linestyle='--')
                            
                            # Add value labels on bars
                            def add_value_labels(bars, ax):
                                for bar in bars:
                                    height = bar.get_height()
                                    if height > 0:  # Only label if height is positive
                                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                               f'${height:,.0f}',
                                               ha='center', va='bottom', fontsize=8, fontweight='bold')
                            
                            add_value_labels(bars1, ax1)
                            add_value_labels(bars2, ax1)
                            
                            # Chart 2: Monthly total comparison for this month
                            # Ensure month is a string
                            month_str = str(month) if not isinstance(month, str) else month
                            monthly_data = pd.DataFrame({
                                'Period': ['Current Month', month_str],
                                'Sales': [float(month_current), float(month_predicted)]
                            })
                            
                            # Ensure all values are proper types
                            monthly_data['Period'] = monthly_data['Period'].astype(str)
                            monthly_data['Sales'] = pd.to_numeric(monthly_data['Sales'], errors='coerce')
                            
                            colors = ['#4CAF50', '#e94560']
                            bars3 = ax2.bar(monthly_data['Period'], monthly_data['Sales'], 
                                          color=colors, alpha=0.8, width=0.6)
                            
                            ax2.set_ylabel('Total Sales ($)', fontsize=12, fontweight='bold')
                            ax2.set_title(f'Monthly Total: Current vs {month_str}', 
                                         fontsize=13, fontweight='bold', pad=15)
                            ax2.grid(axis='y', alpha=0.3, linestyle='--')
                            
                            # Add value labels
                            for bar in bars3:
                                height = bar.get_height()
                                ax2.text(bar.get_x() + bar.get_width()/2., height,
                                       f'${height:,.0f}',
                                       ha='center', va='bottom', fontsize=11, fontweight='bold')
                            
                            # Add change indicator
                            if month_change_pct != 0:
                                arrow_color = '#e94560' if month_change_pct > 0 else '#ff9800'
                                arrow_symbol = '↑' if month_change_pct > 0 else '↓'
                                ax2.text(0.5, max(month_current, month_predicted) * 0.9,
                                       f'{arrow_symbol} {abs(month_change_pct):.1f}% Change',
                                       ha='center', fontsize=12, fontweight='bold', 
                                       color=arrow_color, transform=ax2.transData)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show summary for this month
                            st.markdown(f"#### 📈 Summary for {month_str}")
                            sum_col1, sum_col2, sum_col3 = st.columns(3)
                            with sum_col1:
                                st.metric("Transactions", len(month_df))
                            with sum_col2:
                                st.metric("Current Sales", f"${month_current:,.2f}")
                            with sum_col3:
                                st.metric("Predicted Sales", f"${month_predicted:,.2f}")
                            
                            st.markdown("---")
                        
                        # Display detailed table
                        st.markdown("### Detailed Results")
                        
                        # Select columns to display
                        display_cols = ['Branch', 'City', 'Product line', 'Unit price', 'Quantity', 
                                       'Payment', 'Rating', 'Current Sales', 'Predicted Sales', 
                                       'Difference', 'Error %', 'Base Amount']
                        
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
        
        with col_btn3:
            # Remove last entry button
            if st.button("↩️ Remove Last Entry", use_container_width=True):
                if len(st.session_state.entries) > 0:
                    st.session_state.entries.pop()
                    st.rerun()
    else:
        st.info("👈 Fill in the form above and click 'Add Entry' to start building your transaction list.")
    
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
