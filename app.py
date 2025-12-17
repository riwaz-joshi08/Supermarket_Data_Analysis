"""
Supermarket Sales Prediction App
================================
A beautiful Streamlit application for predicting supermarket sales
using a trained Random Forest model.

Author: AI Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time

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
    /* Import custom fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Headers styling */
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
    
    /* Body text */
    p, label, .stMarkdown {
        font-family: 'Source Sans Pro', sans-serif !important;
        color: #edf2f4 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        border-right: 2px solid #e94560;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #e94560 !important;
        border-bottom: 2px solid #e94560;
        padding-bottom: 10px;
    }
    
    /* Input fields styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background-color: #1a1a2e !important;
        color: #edf2f4 !important;
        border: 1px solid #e94560 !important;
        border-radius: 10px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%) !important;
        color: white !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        padding: 15px 40px !important;
        border-radius: 30px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.6) !important;
    }
    
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        color: #e94560 !important;
        font-size: 3rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #edf2f4 !important;
    }
    
    /* Success/Info boxes */
    .stAlert {
        background-color: rgba(233, 69, 96, 0.1) !important;
        border: 1px solid #e94560 !important;
        border-radius: 15px !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(15, 52, 96, 0.5) !important;
        border-radius: 10px !important;
        color: #e94560 !important;
    }
    
    /* Card container */
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
    
    /* Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #8d99ae;
        font-size: 14px;
        border-top: 1px solid #e94560;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model
# ============================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest pipeline."""
    try:
        model = joblib.load('ML_Model/random_forest_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'ML_Model/random_forest_pipeline.pkl' exists.")
        return None

# ============================================
# Helper Functions
# ============================================
def get_time_of_day(hour):
    """Determine the time of day based on hour."""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def create_input_dataframe(inputs):
    """Create a DataFrame from user inputs with all necessary features."""
    
    # Extract date components
    day_of_week = inputs['date'].weekday()
    day = inputs['date'].day
    month = inputs['date'].month
    hour = inputs['time'].hour
    time_of_day = get_time_of_day(hour)
    
    # Create interaction features
    product_line_time = f"{inputs['product_line']}_{time_of_day}"
    product_line_gender = f"{inputs['product_line']}_{inputs['gender']}"
    branch_time = f"{inputs['branch']}_{time_of_day}"
    
    # Create the input DataFrame
    data = {
        'Branch': [inputs['branch']],
        'City': [inputs['city']],
        'Customer type': [inputs['customer_type']],
        'Gender': [inputs['gender']],
        'Product line': [inputs['product_line']],
        'Unit price': [inputs['unit_price']],
        'Quantity': [inputs['quantity']],
        'Payment': [inputs['payment']],
        'Rating': [inputs['rating']],
        'DayOfWeek': [day_of_week],
        'Day': [day],
        'Month': [month],
        'Hour': [hour],
        'TimeOfDay': [time_of_day],
        'ProductLine_TimeOfDay': [product_line_time],
        'ProductLine_Gender': [product_line_gender],
        'Branch_TimeOfDay': [branch_time]
    }
    
    return pd.DataFrame(data)

# ============================================
# Main Application
# ============================================
def main():
    # Header Section
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1>🛒 Supermarket Sales Predictor</h1>
            <p style="font-size: 1.2em; color: #8d99ae;">
                Predict your sales with AI-powered machine learning
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load the model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar - Input Section
    with st.sidebar:
        st.markdown("## 📝 Enter Transaction Details")
        st.markdown("---")
        
        # Store Information
        st.markdown("### 🏪 Store Information")
        
        branch = st.selectbox(
            "Branch",
            options=["Alex", "Cairo", "Giza"],
            help="Select the store branch"
        )
        
        city = st.selectbox(
            "City",
            options=["Yangon", "Mandalay", "Naypyitaw"],
            help="Select the city location"
        )
        
        st.markdown("---")
        
        # Customer Information
        st.markdown("### 👤 Customer Information")
        
        customer_type = st.selectbox(
            "Customer Type",
            options=["Member", "Normal"],
            help="Member or Normal customer"
        )
        
        gender = st.selectbox(
            "Gender",
            options=["Female", "Male"],
            help="Customer gender"
        )
        
        st.markdown("---")
        
        # Product Information
        st.markdown("### 📦 Product Information")
        
        product_line = st.selectbox(
            "Product Line",
            options=[
                "Health and beauty",
                "Electronic accessories",
                "Home and lifestyle",
                "Sports and travel",
                "Food and beverages",
                "Fashion accessories"
            ],
            help="Select the product category"
        )
        
        unit_price = st.number_input(
            "Unit Price ($)",
            min_value=10.0,
            max_value=100.0,
            value=50.0,
            step=0.01,
            help="Price per unit (10-100)"
        )
        
        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of items (1-10)"
        )
        
        st.markdown("---")
        
        # Transaction Information
        st.markdown("### 💳 Transaction Information")
        
        payment = st.selectbox(
            "Payment Method",
            options=["Ewallet", "Cash", "Credit card"],
            help="Select payment method"
        )
        
        rating = st.slider(
            "Customer Rating",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            help="Customer satisfaction rating (1-10)"
        )
        
        st.markdown("---")
        
        # Date and Time
        st.markdown("### 📅 Date & Time")
        
        transaction_date = st.date_input(
            "Transaction Date",
            value=datetime.now(),
            help="Select the transaction date"
        )
        
        transaction_time = st.time_input(
            "Transaction Time",
            value=time(12, 0),
            help="Select the transaction time"
        )
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Transaction Summary")
        
        # Display input summary in a nice format
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.markdown(f"""
                <div class="info-card">
                    <h4 style="color: #e94560; margin: 0;">🏪 Store</h4>
                    <p style="margin: 5px 0;"><strong>Branch:</strong> {branch}</p>
                    <p style="margin: 5px 0;"><strong>City:</strong> {city}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with summary_col2:
            st.markdown(f"""
                <div class="info-card">
                    <h4 style="color: #e94560; margin: 0;">👤 Customer</h4>
                    <p style="margin: 5px 0;"><strong>Type:</strong> {customer_type}</p>
                    <p style="margin: 5px 0;"><strong>Gender:</strong> {gender}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with summary_col3:
            st.markdown(f"""
                <div class="info-card">
                    <h4 style="color: #e94560; margin: 0;">📦 Product</h4>
                    <p style="margin: 5px 0;"><strong>Line:</strong> {product_line}</p>
                    <p style="margin: 5px 0;"><strong>Payment:</strong> {payment}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Product details
        st.markdown("### 💰 Product Details")
        
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        
        with detail_col1:
            st.metric("Unit Price", f"${unit_price:.2f}")
        
        with detail_col2:
            st.metric("Quantity", f"{quantity}")
        
        with detail_col3:
            st.metric("Rating", f"⭐ {rating}")
        
        with detail_col4:
            base_amount = unit_price * quantity
            st.metric("Base Amount", f"${base_amount:.2f}")
    
    with col2:
        st.markdown("### ⏰ Time Info")
        
        hour = transaction_time.hour
        time_of_day = get_time_of_day(hour)
        
        time_emoji = {
            'Morning': '🌅',
            'Afternoon': '☀️',
            'Evening': '🌆',
            'Night': '🌙'
        }
        
        st.markdown(f"""
            <div class="info-card">
                <p><strong>📅 Date:</strong> {transaction_date.strftime('%B %d, %Y')}</p>
                <p><strong>🕐 Time:</strong> {transaction_time.strftime('%I:%M %p')}</p>
                <p><strong>{time_emoji.get(time_of_day, '⏰')} Period:</strong> {time_of_day}</p>
                <p><strong>📆 Day:</strong> {transaction_date.strftime('%A')}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Prediction Button
    st.markdown("---")
    
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_button = st.button("🔮 Predict Sales", use_container_width=True)
    
    # Make Prediction
    if predict_button:
        with st.spinner('🔄 Analyzing transaction data...'):
            # Prepare inputs
            inputs = {
                'branch': branch,
                'city': city,
                'customer_type': customer_type,
                'gender': gender,
                'product_line': product_line,
                'unit_price': unit_price,
                'quantity': quantity,
                'payment': payment,
                'rating': rating,
                'date': transaction_date,
                'time': transaction_time
            }
            
            # Create input DataFrame
            input_df = create_input_dataframe(inputs)
            
            try:
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display result
                st.markdown("---")
                st.markdown("## 🎯 Prediction Result")
                
                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                
                with result_col2:
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h3 style="color: #8d99ae; margin-bottom: 10px;">Predicted Sales Amount</h3>
                            <h1 style="font-size: 4rem; color: #e94560; margin: 20px 0;" class="pulse-animation">
                                ${prediction:,.2f}
                            </h1>
                            <p style="color: #8d99ae;">
                                Based on the provided transaction details
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 📈 Sales Breakdown")
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    base = unit_price * quantity
                    st.metric("Base Amount (Price × Qty)", f"${base:.2f}")
                
                with insight_col2:
                    tax = prediction * 0.05 / 1.05  # Approximate tax from total
                    st.metric("Estimated Tax (5%)", f"${tax:.2f}")
                
                with insight_col3:
                    cogs = prediction / 1.05  # Cost of goods sold
                    st.metric("Estimated COGS", f"${cogs:.2f}")
                
                # Success message
                st.success("✅ Prediction completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("💡 Please ensure all inputs are filled correctly and the model file is valid.")
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>🛒 Supermarket Sales Predictor | Built with Streamlit & Random Forest ML</p>
            <p>© 2024 | Powered by Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Additional Information Section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
            ### How It Works
            
            This application uses a **Random Forest Regression** model trained on supermarket sales data 
            to predict the total sales amount for a transaction.
            
            #### Features Used for Prediction:
            - **Store Info:** Branch, City
            - **Customer Info:** Customer Type (Member/Normal), Gender
            - **Product Info:** Product Line, Unit Price, Quantity
            - **Transaction Info:** Payment Method, Customer Rating
            - **Temporal Features:** Date components (Day, Month, Day of Week, Hour, Time of Day)
            - **Interaction Features:** Combined features for better predictions
            
            #### Model Performance:
            - **R² Score:** ~99.75% (Cross-validated)
            - **Algorithm:** Random Forest Regressor with 100 estimators
            
            #### Data Preprocessing:
            - Categorical variables are encoded using One-Hot Encoding
            - Numerical features are standardized using Standard Scaler
        """)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
            ### Dataset Details
            
            The model was trained on a supermarket sales dataset containing **1,000 transactions** 
            from three different branches.
            
            #### Branches:
            - 🏪 **Alex** - Yangon (34%)
            - 🏪 **Cairo** - Mandalay (33.2%)
            - 🏪 **Giza** - Naypyitaw (32.8%)
            
            #### Product Lines:
            - 👗 Fashion accessories (17.8%)
            - 🍔 Food and beverages (17.4%)
            - 📱 Electronic accessories (17.0%)
            - ⚽ Sports and travel (16.6%)
            - 🏠 Home and lifestyle (16.0%)
            - 💄 Health and beauty (15.2%)
            
            #### Payment Methods:
            - 📱 E-wallet (34.5%)
            - 💵 Cash (34.4%)
            - 💳 Credit card (31.1%)
        """)

# ============================================
# Run Application
# ============================================
if __name__ == "__main__":
    main()

