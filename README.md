# ❤️ Heart Failure Data Analysis - MEPS Dataset

## 📋 Project Overview

This project performs comprehensive data analysis and predictive modeling on heart failure clinical records using machine learning techniques. The analysis focuses on understanding factors contributing to heart failure outcomes and predicting medical outcomes based on patient characteristics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-yellowgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Dataset Information

### Source
The analysis utilizes the **Heart Failure Clinical Records Dataset**, which contains medical records of patients with heart failure conditions.

### Dataset Structure
The dataset consists of **299 patient records** with **13 clinical features**:

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| `age` | Patient's age | Numerical | 40-95 years |
| `anaemia` | Decrease of red blood cells | Binary | 0: No, 1: Yes |
| `creatinine_phosphokinase` | Level of CPK enzyme | Numerical | 23-7861 IU/L |
| `diabetes` | If patient has diabetes | Binary | 0: No, 1: Yes |
| `ejection_fraction` | Percentage of blood leaving heart | Numerical | 14-80 % |
| `high_blood_pressure` | If patient has hypertension | Binary | 0: No, 1: Yes |
| `platelets` | Platelets in blood | Numerical | 25,000-850,000 |
| `serum_creatinine` | Level of serum creatinine | Numerical | 0.5-9.4 mg/dL |
| `serum_sodium` | Level of serum sodium | Numerical | 114-148 mEq/L |
| `sex` | Patient's gender | Binary | 0: Female, 1: Male |
| `smoking` | If patient smokes | Binary | 0: No, 1: Yes |
| `time` | Follow-up period | Numerical | 4-285 days |
| `DEATH_EVENT` | If patient died during follow-up | Binary | 0: No, 1: Yes |

## 🚀 Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install pandas==1.5.0
pip install numpy==1.23.0
pip install scikit-learn==1.2.0
pip install matplotlib==3.6.0
pip install seaborn==0.12.0
```

### Project Structure
```
heart-failure-analysis/
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── notebooks/
│   └── heart_failure_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── visualization.py
│   ├── model_training.py
│   └── utils.py
├── results/
│   ├── visualizations/
│   └── model_performance/
└── README.md
```

## 📈 Exploratory Data Analysis (EDA)

### Key Visualizations Implemented

#### 1. **Age Distribution**
```python
plt.figure(figsize=(8, 6))
sns.histplot(data['age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```
<img width="494" height="387" alt="image" src="https://github.com/user-attachments/assets/157e0e22-4845-4e5f-9631-af36ef27de68" />


#### 2. **Correlation Matrix**
```python
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

<img width="678" height="602" alt="image" src="https://github.com/user-attachments/assets/b9da58cc-0211-4573-a066-cc91b8fc96f9" />


**Key Insights:**
- Strong positive correlation between `age` and `DEATH_EVENT`
- Negative correlation between `ejection_fraction` and `DEATH_EVENT`
- Positive correlation between `serum_creatinine` and `DEATH_EVENT`

#### 3. **Gender Distribution**
```python
plt.figure(figsize=(6, 6))
data['sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()
```

<img width="340" height="356" alt="image" src="https://github.com/user-attachments/assets/dac38179-8c8a-4e3a-a3e8-446d7e65c7de" />


#### 4. **Clinical Feature Analysis**
- Ejection fraction distribution
- Serum creatinine by diabetes status
- Age vs. Time scatter plot (colored by death event)

## 🤖 Predictive Modeling

### Linear Regression Model

#### Feature Selection
```python
# Features used for prediction
features = [
    'age', 'anaemia', 'creatinine_phosphokinase', 
    'diabetes', 'ejection_fraction', 'high_blood_pressure',
    'platelets', 'serum_creatinine', 'serum_sodium',
    'sex', 'smoking', 'time'
]

# Target variable (using DEATH_EVENT as proxy for medical expenditure)
target = 'DEATH_EVENT'
```

#### Model Training
```python
# Data splitting (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model initialization and training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

#### Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Squared Error (MSE)** | 0.1787 | Lower is better |
| **R-squared (R²)** | 0.2648 | 26.48% variance explained |

## 💰 Healthcare Cost Analysis

### Comparative Analysis Framework
```python
# Create analysis dataset
cost_data = data[[
    'sex', 'age', 'ejection_fraction', 
    'time', 'DEATH_EVENT'
]]

# Group by demographic and clinical factors
cost_summary = cost_data.groupby([
    'sex', 'age', 'ejection_fraction', 'time'
]).mean().reset_index()
```

### Key Findings

#### 1. **Healthcare Costs by Gender**
```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='DEATH_EVENT', data=cost_summary)
plt.title('Healthcare Costs by Sex')
plt.xlabel('Sex (0: Female, 1: Male)')
plt.ylabel('Average Medical Expenditure (DEATH_EVENT)')
plt.show()
```

#### 2. **Risk Factors Analysis**
- **Age**: Strong predictor of adverse outcomes
- **Ejection Fraction**: Lower values correlate with higher mortality
- **Serum Creatinine**: Elevated levels indicate higher risk

## 📊 Statistical Summary

### Dataset Statistics
```python
# Basic statistics
print("Dataset Shape:", data.shape)
print("\nMissing Values:")
print(data.isnull().sum())
print("\nData Types:")
print(data.dtypes)
print("\nBasic Statistics:")
print(data.describe())
```

**Output:**
- **Total Records**: 299
- **Features**: 13
- **Missing Values**: 0
- **Class Distribution**: Imbalanced (203 survived, 96 deceased)

## 🔍 Key Insights

### 1. **Demographic Factors**
- Average patient age: 60.8 years
- Gender distribution: 65% male, 35% female
- Higher mortality observed in older patients

### 2. **Clinical Markers**
- **Ejection Fraction**: Critical indicator (normal range: 50-70%)
- **Serum Creatinine**: Key renal function marker
- **Platelets**: Wide variation observed (25k-850k)

### 3. **Risk Factors**
- **High Risk**: Age > 65, EF < 40%, Serum Creatinine > 1.5
- **Moderate Risk**: Diabetes, Hypertension, Anaemia
- **Lower Impact**: Smoking status showed less correlation

## 🛠️ Usage Examples

### 1. **Data Loading**
```python
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Display basic information
print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
```

### 2. **Quick Analysis**
```python
# Calculate survival rate
survival_rate = data['DEATH_EVENT'].value_counts(normalize=True)
print(f"Survival Rate: {survival_rate[0]*100:.2f}%")
print(f"Mortality Rate: {survival_rate[1]*100:.2f}%")
```

### 3. **Custom Prediction**
```python
def predict_risk(patient_data):
    """
    Predict medical outcome for a new patient
    
    Args:
        patient_data: Dictionary containing patient features
    
    Returns:
        Prediction score (0-1)
    """
    # Convert to DataFrame
    df = pd.DataFrame([patient_data])
    
    # Make prediction
    prediction = model.predict(df)
    return prediction[0]
```

## 📈 Future Enhancements

### Planned Improvements
1. **Advanced Models**
   - Random Forest Classifier
   - Gradient Boosting Machines
   - Neural Networks

2. **Feature Engineering**
   - Create composite risk scores
   - Normalize clinical markers
   - Handle class imbalance

3. **Deployment**
   - Web application for risk prediction
   - API endpoint for integration
   - Real-time monitoring dashboard

4. **Extended Analysis**
   - Time-series analysis of patient progression
   - Cost-effectiveness analysis
   - Treatment outcome comparison

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📚 References

1. Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making.
2. Ahmad, T., et al. (2017). Machine learning methods improve prognostication, identify clinical phenotypes, and suggest heterogeneity in treatment effects for heart failure.
3. MEPS Documentation: [Agency for Healthcare Research and Quality](https://meps.ahrq.gov/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Data Scientist** - Were Vincent Ouma

## 🙏 Acknowledgments

- Medical Expenditure Panel Survey (MEPS) for data collection
- Research institutions for clinical validation
- Open-source community for tools and libraries

---

<div align="center">
  
**📧 Contact**: oumawere2001@gmail.com  

**🌐 Website**: www.vincent.dataupskill.co.ke 

**🐦 Twitter**: @oumawere1

*Last Updated: January 2026*

</div>

---

### ⚠️ Disclaimer
This analysis is for **educational and research purposes only**. It should not be used for clinical decision-making without proper medical supervision. Always consult healthcare professionals for medical advice.
