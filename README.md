# 🛒 Supermarket Sales Prediction Application

A comprehensive, interactive web application for predicting supermarket sales using Machine Learning. Built with **Streamlit** and powered by a **Random Forest Regression** model. The application supports batch predictions, CSV uploads, and provides detailed visualizations and analytics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [How to Use](#-how-to-use)
  - [Method 1: CSV Upload](#method-1-csv-upload)
  - [Method 2: Manual Entry](#method-2-manual-entry)
- [CSV/Excel File Format](#csvexcel-file-format)
- [Understanding the Output](#-understanding-the-output)
- [Model Information](#-model-information)
- [Dataset Information](#-dataset-information)
- [Troubleshooting](#-troubleshooting)
- [Technical Details](#-technical-details)

---

## 🎯 Overview

This application predicts future month sales for supermarket transactions based on various input features. The system supports:

- **Monthly Sales Forecasting**: Predict sales for any target month/year
- **Batch Processing**: Upload CSV files with thousands of transactions or add entries manually
- **Automatic Calculations**: Current sales are automatically calculated based on unit price, quantity, and tax
- **Visual Analytics**: Interactive bar charts comparing current vs predicted sales
- **Smart Insights**: Automatic alerts for predicted sales increases or decreases

The prediction model achieves an impressive **R² score of ~99.75%** using Random Forest Regression.

---

## ✨ Key Features

### 🎨 User Interface
- **Modern Dark Theme**: Beautiful gradient backgrounds with custom styling
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Interactive Elements**: Easy-to-use dropdowns, sliders, and input fields
- **Real-time Feedback**: Instant validation and error messages

### 📊 Data Management
- **CSV Upload**: Bulk import transactions from CSV/Excel files
- **Manual Entry**: Add individual transactions through an intuitive form
- **Tabular Display**: View all entries in an organized table format
- **Entry Management**: Add, remove, or clear entries with simple buttons

### 🔮 Prediction Capabilities
- **Monthly Forecasting**: Select any target month/year for predictions
- **Batch Predictions**: Process multiple transactions simultaneously
- **Automatic Sales Calculation**: Current sales computed as `Unit Price × Quantity × 1.05`
- **Comparison Metrics**: View current vs predicted sales with difference and error percentages

### 📈 Visualizations & Analytics
- **Product Line Charts**: Bar graphs showing sales by product category
- **Monthly Comparison**: Side-by-side comparison of current month vs target month
- **Summary Metrics**: Total transactions, sales amounts, and average error
- **Trend Indicators**: Visual arrows and color coding for sales changes

### 💡 Smart Features
- **Auto-Retrain**: Automatically retrains model if not fitted
- **Data Validation**: Handles missing values and type conversions
- **Error Handling**: Graceful error messages with detailed tracebacks
- **Export Results**: Download predictions as CSV files

---

## 📁 Project Structure

```
Supermarket_Data_Analysis/
│
├── app.py                          # Main Streamlit application
├── README.md                       # This documentation file
├── requirements.txt                # Python dependencies
│
├── Data_Sets/
│   └── SuperMarket Analysis.csv   # Original training dataset
│
├── ML_Model/
│   └── random_forest_pipeline.pkl # Trained ML model pipeline
│
├── Supermarket_Data_Analysis.ipynb # Jupyter notebook with EDA & model training
│
└── env/                            # Virtual environment (optional)
```

---

## 📋 Prerequisites

Before running the application, ensure you have the following installed:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git** (optional, for cloning repositories)

---

## 🚀 Installation

### Step 1: Navigate to the Project Directory

```bash
cd C:\Users\Lenovo\OneDrive\Desktop\Supermarket_Data_Analysis
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv env
.\env\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python -m venv env
source env/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

---

## ▶️ Running the Application

### Method 1: Using Streamlit Command (Recommended)

1. **Open Terminal/Command Prompt**

2. **Navigate to the project directory:**
   ```bash
   cd C:\Users\Lenovo\OneDrive\Desktop\Supermarket_Data_Analysis
   ```

3. **Activate the virtual environment (if using):**
   ```powershell
   .\env\Scripts\Activate.ps1
   ```

4. **Run the Streamlit app:**
```bash
streamlit run app.py
```

5. **Access the application:**
   - The app will automatically open in your default web browser
   - If not, manually navigate to: **http://localhost:8501**

### Method 2: Using Python Directly

```bash
python -m streamlit run app.py
```

### Stopping the Application

- Press `Ctrl + C` in the terminal to stop the server

---

## 📖 How to Use

### Step 1: Select Target Month for Prediction

At the top of the application, you'll find:
- **Target Year**: Select the year for prediction (2020-2030)
- **Target Month**: Select the month (January through December)

This determines which month the model will predict sales for.

### Method 1: CSV Upload

1. **Prepare Your CSV File**
   - Ensure your CSV follows the format specified in [CSV/Excel File Format](#csvexcel-file-format)
   - Click on the expandable section "📋 CSV/Excel File Format Requirements" for detailed specifications

2. **Upload the File**
   - Click "Browse files" under "📤 Upload CSV (Optional)"
   - Select your CSV file
   - The application will automatically:
     - Parse all rows
     - Calculate current sales (if not provided)
     - Add entries to the transaction table
     - Display a success message with the number of loaded entries

3. **Review Entries**
   - All uploaded entries will appear in the "📊 Transaction Entries" table
   - Verify the data is correct

4. **Generate Predictions**
   - Click the "🔮 Predict Sales" button
   - Wait for processing (may take a few seconds for large files)
   - View results with visualizations and detailed metrics

### Method 2: Manual Entry

1. **Fill in Transaction Details**
   
   **Store Information:**
   - **Branch**: Select from Alex, Cairo, or Giza
   - **City**: Select from Yangon, Mandalay, or Naypyitaw
   
   **Customer Information:**
   - **Customer Type**: Member or Normal
   - **Gender**: Female or Male
   
   **Product Information:**
   - **Product Line**: Choose from available categories
   - **Unit Price**: Enter value between $10.00 - $100.00
   - **Quantity**: Enter number between 1-10
   
   **Transaction Information:**
   - **Payment Method**: Ewallet, Cash, or Credit card
   - **Customer Rating**: Use slider (1.0 - 10.0)
   - **Transaction Date**: Select date (defaults to target month)
   - **Transaction Time**: Select time

2. **Review Calculated Sales**
   - The application automatically calculates current sales
   - Formula: `Unit Price × Quantity × 1.05` (includes 5% tax)
   - Displayed as: "💰 Calculated Current Sales: $XXX.XX"

3. **Add Entry**
   - Click the "➕ Add Entry" button
   - The entry will be added to the transaction table
   - You can add multiple entries before predicting

4. **Manage Entries**
   - **View Table**: All entries are displayed in "📊 Transaction Entries"
   - **Remove Last Entry**: Click "↩️ Remove Last Entry"
   - **Clear All**: Click "🗑️ Clear All Entries"

5. **Generate Predictions**
   - Once you have entries in the table, click "🔮 Predict Sales"
   - Results will be displayed with charts and detailed metrics

---

## 📄 CSV/Excel File Format

### Required Columns

Your CSV file must include the following columns (case-sensitive):

| Column Name | Data Type | Example Values | Required |
|------------|-----------|----------------|----------|
| Branch | Text | Alex, Cairo, Giza | ✅ Yes |
| City | Text | Yangon, Mandalay, Naypyitaw | ✅ Yes |
| Customer type | Text | Member, Normal | ✅ Yes |
| Gender | Text | Female, Male | ✅ Yes |
| Product line | Text | Health and beauty, Electronic accessories, Home and lifestyle, Sports and travel, Food and beverages, Fashion accessories | ✅ Yes |
| Unit price | Number | 10.0 - 100.0 | ✅ Yes |
| Quantity | Integer | 1 - 10 | ✅ Yes |
| Payment | Text | Ewallet, Cash, Credit card | ✅ Yes |
| Rating | Number | 1.0 - 10.0 | ✅ Yes |
| Date | Date | 1/5/2019, 2019-01-05 | ✅ Yes |
| Time | Time | 1:08:00 PM, 13:08:00 | ✅ Yes |

### Optional Columns

| Column Name | Data Type | Example Values | Required |
|------------|-----------|----------------|----------|
| Target Month | Text | January 2024 | ❌ No |
| Current Sales | Number | 548.97 | ❌ No |

### Format Specifications

- **Date Format**: 
  - `MM/DD/YYYY` (e.g., `1/5/2019`)
  - `YYYY-MM-DD` (e.g., `2019-01-05`)
  
- **Time Format**:
  - `HH:MM:SS AM/PM` (e.g., `1:08:00 PM`)
  - `HH:MM:SS` (24-hour format, e.g., `13:08:00`)

- **Text Values**: All text values are **case-sensitive**
  - ✅ Correct: `Member`, `Female`, `Ewallet`
  - ❌ Incorrect: `member`, `female`, `ewallet`

- **Calculations**:
  - If `Current Sales` is not provided, it will be calculated as: `Unit price × Quantity × 1.05`
  - If `Target Month` is not provided, entries will use the selected month from the form

### Example CSV Structure

```csv
Branch,City,Customer type,Gender,Product line,Unit price,Quantity,Payment,Rating,Date,Time
Alex,Yangon,Member,Female,Health and beauty,50.0,5,Ewallet,7.5,2019-01-05,13:08:00
Cairo,Mandalay,Normal,Male,Electronic accessories,75.0,3,Cash,8.0,2019-01-06,14:30:00
Giza,Naypyitaw,Member,Female,Food and beverages,30.0,7,Credit card,9.0,2019-01-07,10:15:00
```

---

## 📊 Understanding the Output

### Summary Metrics

After clicking "🔮 Predict Sales", you'll see:

1. **Total Transactions**: Number of transactions processed
2. **Current Month Sales**: Sum of all current sales amounts
3. **Predicted Sales (Target Month)**: Sum of all predicted sales for the target month
4. **Avg Error %**: Average absolute percentage error across all predictions

### Monthly Sales Projection

Two cards display:
- **Current Month Total**: Sum of current sales
- **Target Month Projection**: Predicted total sales with change percentage

### Visualizations

#### 1. Sales by Product Line Chart
- **Left Chart**: Bar graph comparing current vs predicted sales for each product line
- Shows sales distribution across different product categories
- Color-coded: Green (Current) vs Red (Predicted)

#### 2. Monthly Total Comparison Chart
- **Right Chart**: Side-by-side comparison of total current month vs target month
- Includes percentage change indicator with arrow (↑ for increase, ↓ for decrease)

### Conditional Messages

The application automatically displays messages based on predictions:

- **⚠️ Sales Decrease Predicted**: If predicted sales < current sales
  - Shows decrease amount and percentage
  - Warning-style message

- **📈 Sales Increase Predicted**: If predicted sales > current sales
  - Shows increase amount and percentage
  - Success-style message

- **➡️ Sales Stable**: If predicted sales ≈ current sales
  - Info-style message

### Detailed Results Table

A comprehensive table showing:
- Transaction details (Branch, City, Product line, etc.)
- **Current Sales**: Calculated or provided sales amount
- **Predicted Sales**: Model's prediction
- **Difference**: `Predicted Sales - Current Sales`
- **Error %**: `((Difference / Current Sales) × 100)`
- **Base Amount**: `Unit Price × Quantity`

### Export Results

- Click "📥 Download Results as CSV" to save all predictions
- File name format: `sales_predictions_YYYYMMDD_HHMMSS.csv`

---

## 🤖 Model Information

### Algorithm
- **Random Forest Regressor** with 100 estimators
- **Random State**: 42 (for reproducibility)

### Performance Metrics

| Metric | Score |
|--------|-------|
| R² Score | 99.75% |
| Cross-Validation R² | 99.77% |
| Mean Absolute Error | 8.52 |

### Features Used

The model uses the following features for prediction:

#### Categorical Features
- Branch (Alex, Cairo, Giza)
- City (Yangon, Mandalay, Naypyitaw)
- Customer Type (Member, Normal)
- Gender (Female, Male)
- Product Line (6 categories)
- Payment Method (Ewallet, Cash, Credit card)
- TimeOfDay (Morning, Afternoon, Evening, Night)

#### Numerical Features
- Unit Price
- Quantity
- Rating
- DayOfWeek (0-6, Monday-Sunday)
- Day (1-31)
- Month (1-12)
- Hour (0-23)

#### Interaction Features
- `ProductLine_TimeOfDay`: Product line × Time of day
- `ProductLine_Gender`: Product line × Gender
- `Branch_TimeOfDay`: Branch × Time of day

### Preprocessing Pipeline

1. **Feature Engineering**:
   - Extract date/time features (DayOfWeek, Day, Month, Hour)
   - Create TimeOfDay categories (Morning, Afternoon, Evening, Night)
   - Generate interaction features

2. **Categorical Encoding**: One-Hot Encoding (handle_unknown="ignore")
3. **Numerical Scaling**: Standard Scaler (mean=0, std=1)
4. **Column Transformation**: Applied via `ColumnTransformer`

### Model Architecture

```
Pipeline:
├── Preprocessing (ColumnTransformer)
│   ├── Categorical → OneHotEncoder
│   └── Numerical → StandardScaler
└── Model (RandomForestRegressor)
    └── n_estimators=100
```

### Auto-Retrain Mechanism

If the model is not fitted when loaded, the application will:
1. Load the training data from `Data_Sets/SuperMarket Analysis.csv`
2. Apply the same preprocessing steps
3. Fit the pipeline
4. Save the retrained model
5. Display a success message

---

## 📊 Dataset Information

The model was trained on the **SuperMarket Analysis** dataset containing:

- **1,000 transactions**
- **17 original features**
- **3 store branches** (Alex, Cairo, Giza)
- **3 cities** (Yangon, Mandalay, Naypyitaw)

### Feature Distribution

| Category | Distribution |
|----------|--------------|
| **Branches** | Alex (34%), Cairo (33.2%), Giza (32.8%) |
| **Customer Type** | Member (56.5%), Normal (43.5%) |
| **Gender** | Female (57.1%), Male (42.9%) |
| **Payment** | Ewallet (34.5%), Cash (34.4%), Credit card (31.1%) |

### Product Lines

- Fashion accessories (17.8%)
- Food and beverages (17.4%)
- Electronic accessories (17.0%)
- Sports and travel (16.6%)
- Home and lifestyle (16.0%)
- Health and beauty (15.2%)

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Model file not found" Error

**Error Message:**
```
❌ Error loading model: [Errno 2] No such file or directory
```

**Solution:**
- Ensure the model file exists at: `ML_Model/random_forest_pipeline.pkl`
- If missing, the app will attempt to retrain using `Data_Sets/SuperMarket Analysis.csv`
- Ensure the training data file exists

#### 2. "Training data not found" Error

**Error Message:**
```
❌ Training data not found! Please ensure 'Data_Sets/SuperMarket Analysis.csv' exists.
```

**Solution:**
- Verify the file exists at: `Data_Sets/SuperMarket Analysis.csv`
- Check file name spelling (case-sensitive)
- Ensure the file is not corrupted

#### 3. "ModuleNotFoundError"

**Error Message:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

#### 4. Streamlit not recognized

**Error Message:**
```
'streamlit' is not recognized as an internal or external command
```

**Solution:**
- Ensure Streamlit is installed: `pip install streamlit`
- Use full path: `python -m streamlit run app.py`
- Activate virtual environment if using one

#### 5. Virtual environment issues

**Error Message:**
```
The term 'Activate.ps1' is not recognized
```

**Solution (Windows PowerShell):**
```powershell
# Set execution policy (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.\env\Scripts\Activate.ps1
```

**Alternative (Command Prompt):**
```cmd
.\env\Scripts\activate.bat
```

#### 6. Port already in use

**Error Message:**
```
Port 8501 is already in use
```

**Solution:**
```bash
# Run on a different port
streamlit run app.py --server.port 8502
```

#### 7. CSV Upload Errors

**Error Message:**
```
❌ Error loading CSV: ...
```

**Common Causes:**
- Missing required columns
- Incorrect data types
- Invalid date/time formats
- Case-sensitive text values don't match

**Solution:**
- Review the [CSV/Excel File Format](#csvexcel-file-format) section
- Ensure all required columns are present
- Check that text values match exactly (case-sensitive)
- Verify date/time formats are correct

#### 8. "Pipeline is not fitted yet" Error

**Error Message:**
```
NotFittedError: Pipeline is not fitted yet.
```

**Solution:**
- The app should automatically retrain the model
- If it doesn't, check that `Data_Sets/SuperMarket Analysis.csv` exists
- Manually retrain by running the Jupyter notebook: `Supermarket_Data_Analysis.ipynb`

#### 9. Type Errors in Visualizations

**Error Message:**
```
TypeError: 'value' must be an instance of str or bytes, not a float
```

**Solution:**
- This is usually caused by NaN values in string columns
- The app handles this automatically, but if it persists:
  - Check your CSV for missing values
  - Ensure "Target Month" column is text type
  - Verify all categorical columns are strings

---

## 🔬 Technical Details

### Architecture

- **Framework**: Streamlit (Python web framework)
- **ML Library**: scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib

### Key Functions

#### `load_model()`
- Loads the trained pipeline from disk
- Checks if model is fitted
- Triggers auto-retrain if needed
- Returns model and pipeline objects

#### `retrain_model()`
- Loads training data
- Applies feature engineering
- Creates and fits preprocessing pipeline
- Trains Random Forest model
- Saves the fitted pipeline

#### `preprocess_data(df)`
- Replicates preprocessing from training notebook
- Extracts date/time features
- Creates interaction features
- Handles missing values
- Returns processed DataFrame ready for prediction

#### `get_time_of_day(hour)`
- Categorizes hour into time periods:
  - Morning: 5:00 - 11:59
  - Afternoon: 12:00 - 16:59
  - Evening: 17:00 - 20:59
  - Night: 21:00 - 4:59

### Session State Management

The app uses Streamlit's `st.session_state` to maintain:
- **entries**: List of transaction entries
- Persists across reruns
- Allows adding/removing entries without losing data

### Performance Considerations

- **Caching**: Model loading is cached using `@st.cache_resource`
- **Batch Processing**: Handles large CSV files efficiently
- **Memory Management**: Processes data in chunks when needed

### Data Flow

```
User Input (CSV/Form)
    ↓
Data Validation & Parsing
    ↓
Feature Engineering (preprocess_data)
    ↓
Pipeline Preprocessing (ColumnTransformer)
    ↓
Model Prediction (RandomForestRegressor)
    ↓
Post-processing & Visualization
    ↓
Results Display & Export
```

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the Jupyter notebook (`Supermarket_Data_Analysis.ipynb`) for model details
3. Ensure all dependencies are correctly installed
4. Verify your CSV file format matches the specifications

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🙏 Acknowledgments

- **Dataset**: Supermarket Sales Analysis Dataset
- **Framework**: Streamlit
- **ML Library**: scikit-learn
- **Visualization**: Matplotlib, Seaborn

---

## 🚀 Future Enhancements

Potential improvements for future versions:

- [ ] Support for Excel (.xlsx) file uploads
- [ ] Real-time prediction updates
- [ ] Advanced filtering and search in results table
- [ ] Export to PDF reports
- [ ] Historical trend analysis
- [ ] Multi-month forecasting
- [ ] Interactive dashboard with more chart types
- [ ] User authentication and data persistence

---

**Happy Predicting! 🛒📈**

---

*Last Updated: Based on current system implementation*
