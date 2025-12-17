# 🛒 Supermarket Sales Prediction Application

A beautiful, interactive web application for predicting supermarket sales using Machine Learning. Built with **Streamlit** and powered by a **Random Forest Regression** model.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [How to Use](#-how-to-use)
- [Model Information](#-model-information)
- [Dataset Information](#-dataset-information)
- [Troubleshooting](#-troubleshooting)
- [Screenshots](#-screenshots)

---

## 🎯 Overview

This application predicts the total sales amount for a supermarket transaction based on various input features such as:

- Store branch and location
- Customer demographics
- Product details
- Transaction timing
- Payment method

The prediction model achieves an impressive **R² score of ~99.75%** using Random Forest Regression.

---

## ✨ Features

- 🎨 **Beautiful UI** - Modern, dark-themed interface with gradient backgrounds
- 📊 **Real-time Predictions** - Instant sales predictions based on input data
- 📱 **Responsive Design** - Works on desktop and mobile devices
- 🔄 **Interactive Inputs** - Easy-to-use dropdowns, sliders, and input fields
- 📈 **Sales Breakdown** - Detailed breakdown of predicted sales components
- ℹ️ **Informative Tooltips** - Helpful hints for each input field

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
│   └── SuperMarket Analysis.csv    # Original dataset
│
├── ML_Model/
│   └── random_forest_pipeline.pkl  # Trained ML model pipeline
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

---

## 🚀 Installation

### Step 1: Clone or Navigate to the Project Directory

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

### Step 1: Enter Store Information
- Select the **Branch** (Alex, Cairo, or Giza)
- Select the **City** (Yangon, Mandalay, or Naypyitaw)

### Step 2: Enter Customer Information
- Choose **Customer Type** (Member or Normal)
- Select **Gender** (Female or Male)

### Step 3: Enter Product Information
- Select **Product Line** from available categories
- Enter **Unit Price** ($10 - $100)
- Enter **Quantity** (1-10 items)

### Step 4: Enter Transaction Information
- Select **Payment Method** (Ewallet, Cash, or Credit card)
- Set **Customer Rating** (1-10 scale)

### Step 5: Set Date and Time
- Choose **Transaction Date**
- Set **Transaction Time**

### Step 6: Get Prediction
- Click the **"🔮 Predict Sales"** button
- View the predicted sales amount and breakdown

---

## 🤖 Model Information

### Algorithm
- **Random Forest Regressor** with 100 estimators

### Performance Metrics
| Metric | Score |
|--------|-------|
| R² Score | 99.75% |
| Cross-Validation R² | 99.77% |
| Mean Absolute Error | 8.52 |

### Features Used
The model uses the following features for prediction:

| Feature Type | Features |
|--------------|----------|
| **Categorical** | Branch, City, Customer Type, Gender, Product Line, Payment, TimeOfDay |
| **Numerical** | Unit Price, Quantity, Rating, DayOfWeek, Day, Month, Hour |
| **Interaction** | ProductLine_TimeOfDay, ProductLine_Gender, Branch_TimeOfDay |

### Preprocessing Pipeline
1. **Categorical Encoding:** One-Hot Encoding
2. **Numerical Scaling:** Standard Scaler

---

## 📊 Dataset Information

The model was trained on the **SuperMarket Analysis** dataset containing:

- **1,000 transactions**
- **17 original features**
- **3 store branches**

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
**Solution:** Ensure the model file exists at:
```
ML_Model/random_forest_pipeline.pkl
```

#### 2. "ModuleNotFoundError" 
**Solution:** Install missing dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Streamlit not recognized
**Solution:** Ensure Streamlit is installed:
```bash
pip install streamlit
```

#### 4. Virtual environment issues
**Solution:** Create a new virtual environment:
```bash
python -m venv new_env
.\new_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 5. Port already in use
**Solution:** Run on a different port:
```bash
streamlit run app.py --server.port 8502
```

#### 6. PowerShell script execution policy error
**Solution:** Run this command in PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📸 Screenshots

### Main Interface
The application features a beautiful dark theme with:
- Gradient backgrounds
- Custom styled inputs
- Interactive prediction cards

### Sidebar
- All input fields organized in sections
- Helpful tooltips and descriptions

### Prediction Results
- Large, prominent prediction display
- Sales breakdown with tax and COGS estimates
- Success/error notifications

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the Jupyter notebook for model details
3. Ensure all dependencies are correctly installed

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🙏 Acknowledgments

- Dataset: Supermarket Sales Analysis Dataset
- Framework: Streamlit
- ML Library: scikit-learn

---

**Happy Predicting! 🛒📈**
