import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Supermarket Sales Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin-top: 2rem;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    try:
        model_path = os.path.join('ML_Model', 'GradientBoostingRegressor.pkl')
        preprocessor_path = os.path.join('ML_Model', 'preprocessor.pkl')
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {str(e)}")
        st.stop()

# Feature engineering function
def create_features(data):
    """Create engineered features matching the training pipeline"""
    df = data.copy()
    
    # Extract Item_Category from Item_ID (first 2 characters)
    item_category_map = {
        'FD': 'Food',
        'DR': 'Drinks',
        'NC': 'Non-Consumable'
    }
    df['Item_Category'] = df['Item_ID'].str[:2].map(item_category_map)
    
    # Handle invalid Item_ID format - default to 'Food' if mapping fails
    if df['Item_Category'].isna().any():
        df['Item_Category'] = df['Item_Category'].fillna('Food')
    
    # Create MRP_Weight interaction feature
    df['MRP_Weight'] = df['Item_MRP'] * df['Item_W']
    
    # Create one-hot encoded Item_Type columns
    # List of all Item_Type columns in the exact order expected by the model
    # Note: 'Baking Goods' is the reference category (dropped with drop_first=True)
    item_type_columns = [
        'Item_Type_Breads', 'Item_Type_Breakfast', 'Item_Type_Canned', 
        'Item_Type_Dairy', 'Item_Type_Frozen Foods', 'Item_Type_Fruits and Vegetables',
        'Item_Type_Hard Drinks', 'Item_Type_Health and Hygiene', 'Item_Type_Household',
        'Item_Type_Meat', 'Item_Type_Others', 'Item_Type_Seafood', 
        'Item_Type_Snack Foods', 'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods'
    ]
    
    # Initialize all Item_Type columns to 0
    for col in item_type_columns:
        df[col] = 0
    
    # Set the appropriate Item_Type column to 1 based on input
    item_type_mapping = {
        'Baking Goods': None,  # Reference category (dropped)
        'Breads': 'Item_Type_Breads',
        'Breakfast': 'Item_Type_Breakfast',
        'Canned': 'Item_Type_Canned',
        'Dairy': 'Item_Type_Dairy',
        'Frozen Foods': 'Item_Type_Frozen Foods',
        'Fruits and Vegetables': 'Item_Type_Fruits and Vegetables',
        'Hard Drinks': 'Item_Type_Hard Drinks',
        'Health and Hygiene': 'Item_Type_Health and Hygiene',
        'Household': 'Item_Type_Household',
        'Meat': 'Item_Type_Meat',
        'Others': 'Item_Type_Others',
        'Seafood': 'Item_Type_Seafood',
        'Snack Foods': 'Item_Type_Snack Foods',
        'Soft Drinks': 'Item_Type_Soft Drinks',
        'Starchy Foods': 'Item_Type_Starchy Foods'
    }
    
    if df['Item_Type'].iloc[0] in item_type_mapping and item_type_mapping[df['Item_Type'].iloc[0]]:
        df[item_type_mapping[df['Item_Type'].iloc[0]]] = 1
    
    # Features that depend on Sales or Outlet_Year (not available at prediction time)
    # Set to 0 - preprocessor will handle imputation if needed
    df['MRP_Sales_interaction'] = 0
    df['MRP_OutletYear_interaction'] = 0
    df['Sales_MRPWeight_interaction'] = 0
    
    # Drop columns that are not used in the model
    columns_to_drop = ['Item_ID', 'Item_Type']  # Item_Type is now one-hot encoded
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Ensure columns are in the exact order expected by the preprocessor
    expected_columns = [
        'Item_W', 'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Item_Category',
        'MRP_Weight', 'MRP_Sales_interaction', 'MRP_OutletYear_interaction', 
        'Sales_MRPWeight_interaction'
    ] + item_type_columns
    
    # Reorder columns to match expected order
    df = df[[col for col in expected_columns if col in df.columns]]
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🛒 Supermarket Sales Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()
    
    # Sidebar for input
    with st.sidebar:
        st.header("📋 Input Parameters")
        st.markdown("Enter the product and outlet details below:")
        
        # Item_ID input
        item_id = st.text_input(
            "Item ID",
            value="FDX01",
            help="Enter the Item ID (e.g., FDX01, DRX02, NCX03). First 2 characters determine category: FD=Food, DR=Drinks, NC=Non-Consumable"
        )
        
        # Item_W input
        item_w = st.number_input(
            "Item Weight (Item_W)",
            min_value=0.0,
            value=12.0,
            step=0.1,
            format="%.2f",
            help="Weight of the item"
        )
        
        # Item_Type input
        item_type = st.selectbox(
            "Item Type",
            options=[
                'Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy',
                'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks',
                'Health and Hygiene', 'Household', 'Meat', 'Others',
                'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'
            ],
            index=0,
            help="Type of the item. Note: 'Baking Goods' is the reference category."
        )
        
        # Item_MRP input
        item_mrp = st.number_input(
            "Item MRP (Maximum Retail Price)",
            min_value=0.0,
            value=100.0,
            step=0.1,
            format="%.2f",
            help="Maximum Retail Price of the item"
        )
        
        # Outlet_Size input
        outlet_size = st.selectbox(
            "Outlet Size",
            options=['Small', 'Medium', 'High'],
            index=0,
            help="Size of the outlet"
        )
        
        # Outlet_Location_Type input
        outlet_location_type = st.selectbox(
            "Outlet Location Type",
            options=['Tier 1', 'Tier 2', 'Tier 3'],
            index=0,
            help="Location tier of the outlet"
        )
        
        st.markdown("---")
        predict_button = st.button("🔮 Predict Sales", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Prediction Details")
        
        if predict_button:
            try:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'Item_ID': [item_id],
                    'Item_W': [item_w],
                    'Item_Type': [item_type],
                    'Item_MRP': [item_mrp],
                    'Outlet_Size': [outlet_size],
                    'Outlet_Location_Type': [outlet_location_type]
                })
                
                # Feature engineering
                processed_data = create_features(input_data)
                
                # Display processed features (for debugging/information)
                with st.expander("🔍 View Processed Features"):
                    st.dataframe(processed_data, use_container_width=True)
                
                # Preprocess the data
                processed_array = preprocessor.transform(processed_data)
                
                # Make prediction
                prediction = model.predict(processed_array)[0]
                
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### Predicted Sales")
                st.markdown(f'<div class="prediction-value">${prediction:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional information
                st.info(f"💡 The model predicts sales of **${prediction:,.2f}** based on the provided input parameters.")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.exception(e)
        
        else:
            st.info("👈 Please fill in the input parameters in the sidebar and click 'Predict Sales' to get a prediction.")
    
    with col2:
        st.subheader("ℹ️ About")
        st.markdown("""
        This application uses a **Gradient Boosting Regressor** model 
        to predict supermarket sales based on:
        
        - **Item characteristics**: ID, Weight, Type, MRP
        - **Outlet characteristics**: Size, Location Type
        
        The model has been trained on historical sales data and 
        uses advanced feature engineering techniques to provide 
        accurate predictions.
        """)
        
        st.markdown("---")
        st.subheader("📝 Input Guidelines")
        st.markdown("""
        - **Item ID**: Use format like FDX01 (Food), DRX02 (Drinks), NCX03 (Non-Consumable)
        - **Item Weight**: Enter weight in appropriate units
        - **Item Type**: Select from the dropdown menu
        - **Item MRP**: Enter the maximum retail price
        - **Outlet Size**: Choose Small, Medium, or High
        - **Outlet Location**: Select Tier 1, 2, or 3
        """)

if __name__ == "__main__":
    main()

