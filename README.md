# Supermarket Sales Prediction App

A beautiful and interactive Streamlit web application for predicting supermarket sales based on product and outlet characteristics. This application uses a trained Gradient Boosting Regressor model to provide accurate sales predictions.

## 🎯 Features

- **Interactive Web Interface**: User-friendly Streamlit-based frontend
- **Real-time Predictions**: Get instant sales predictions based on input parameters
- **Feature Engineering**: Automatic feature engineering matching the training pipeline
- **Model Integration**: Uses pre-trained Gradient Boosting Regressor model
- **Beautiful UI**: Modern and responsive design with custom styling

## 📋 Prerequisites

Before running the application, ensure you have the following:

1. **Python 3.7 or higher** installed on your system
2. **Required model files** in the `ML_Model/` directory:
   - `GradientBoostingRegressor.pkl` - The trained model
   - `preprocessor.pkl` - The preprocessing pipeline

## 🚀 Installation

### Step 1: Clone or Navigate to the Project Directory

```bash
cd Supermarket_Data_Analysis
```

### Step 2: Create a Virtual Environment (Recommended)

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

## 📦 Required Files Structure

Ensure your project directory has the following structure:

```
Supermarket_Data_Analysis/
│
├── app.py                          # Streamlit application
├── ML_Model/
│   ├── GradientBoostingRegressor.pkl   # Trained model
│   └── preprocessor.pkl               # Preprocessing pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🏃 Running the Application

### Method 1: Using Streamlit Command

Once you have installed all dependencies and activated your virtual environment, run:

```bash
streamlit run app.py
```

### Method 2: Using Python Module

```bash
python -m streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`. If it doesn't open automatically, you can manually navigate to this URL.

## 📖 How to Use the Application

1. **Launch the Application**: Run `streamlit run app.py` as described above

2. **Fill in the Input Parameters** (in the sidebar):
   - **Item ID**: Enter the item identifier (e.g., `FDX01`, `DRX02`, `NCX03`)
     - `FD` prefix = Food category
     - `DR` prefix = Drinks category
     - `NC` prefix = Non-Consumable category
   
   - **Item Weight (Item_W)**: Enter the weight of the item (numeric value)
   
   - **Item Type**: Select from the dropdown menu (e.g., Baking Goods, Breads, Dairy, etc.)
   
   - **Item MRP**: Enter the Maximum Retail Price of the item
   
   - **Outlet Size**: Select from Small, Medium, or High
   
   - **Outlet Location Type**: Select from Tier 1, Tier 2, or Tier 3

3. **Click "Predict Sales"**: The application will:
   - Process your inputs through feature engineering
   - Apply the preprocessing pipeline
   - Generate a sales prediction using the trained model
   - Display the predicted sales value

4. **View Results**: The predicted sales amount will be displayed prominently in the main area

## 🔧 Input Field Guidelines

### Item ID Format
- Use format like: `FDX01`, `DRX02`, `NCX03`
- First 2 characters determine the category:
  - `FD` = Food
  - `DR` = Drinks  
  - `NC` = Non-Consumable

### Item Type Options
The following item types are available:
- Baking Goods (reference category)
- Breads
- Breakfast
- Canned
- Dairy
- Frozen Foods
- Fruits and Vegetables
- Hard Drinks
- Health and Hygiene
- Household
- Meat
- Others
- Seafood
- Snack Foods
- Soft Drinks
- Starchy Foods

### Outlet Size
- Small
- Medium
- High

### Outlet Location Type
- Tier 1
- Tier 2
- Tier 3

## 🛠️ Technical Details

### Model Information
- **Algorithm**: Gradient Boosting Regressor
- **Training**: Model trained on historical supermarket sales data
- **Preprocessing**: Includes feature engineering, imputation, scaling, and encoding

### Feature Engineering
The application automatically performs the following feature engineering:
- Extracts `Item_Category` from `Item_ID` (first 2 characters)
- Creates `MRP_Weight` interaction feature (Item_MRP × Item_W)
- One-hot encodes `Item_Type` into multiple binary columns
- Handles interaction features that depend on unavailable data (set to 0)

### Preprocessing Pipeline
The preprocessor handles:
- Numeric feature imputation (median strategy)
- Standard scaling of numeric features
- Categorical feature imputation (most frequent strategy)
- One-hot encoding of categorical features

## 🐛 Troubleshooting

### Issue: "Error loading model or preprocessor"
**Solution**: 
- Ensure the `ML_Model/` directory exists
- Verify that both `GradientBoostingRegressor.pkl` and `preprocessor.pkl` are present
- Check that the files are not corrupted

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: 
- Install streamlit: `pip install streamlit`
- Ensure your virtual environment is activated

### Issue: Port already in use
**Solution**: 
- Streamlit will automatically try the next available port
- Or specify a different port: `streamlit run app.py --server.port 8502`

### Issue: Prediction errors or unexpected results
**Solution**:
- Verify all input fields are filled correctly
- Check that Item ID format matches expected pattern (FD/DR/NC prefix)
- Ensure numeric inputs are positive values

## 📝 Dependencies

The application requires the following Python packages:
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning library
- `joblib` - Model serialization

All dependencies are listed in `requirements.txt`.

## 🔄 Updating the Model

If you retrain the model and want to use the new version:

1. Replace `ML_Model/GradientBoostingRegressor.pkl` with your new model
2. Replace `ML_Model/preprocessor.pkl` with your new preprocessor
3. Ensure the feature engineering in `app.py` matches your training pipeline
4. Restart the Streamlit application

## 📊 Model Performance

The Gradient Boosting Regressor model was selected based on cross-validation performance. For details on model training and evaluation, refer to the `Supermarket_Data_Analysis.ipynb` notebook.

## 🤝 Contributing

If you want to improve the application:
1. Ensure feature engineering matches the training pipeline
2. Test with various input combinations
3. Maintain the same input field structure
4. Update this README if you add new features

## 📄 License

This project is part of a data analysis and machine learning workflow for supermarket sales prediction.

## 💡 Tips

- Use realistic values for Item Weight and MRP based on typical supermarket products
- The Item ID format is important - ensure it starts with FD, DR, or NC
- Experiment with different combinations to see how predictions change
- The "View Processed Features" expander shows the engineered features for debugging

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all files are in the correct locations
3. Ensure all dependencies are installed
4. Check that Python version is 3.7 or higher

---

**Happy Predicting! 🎉**

