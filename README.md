# Predicting Chronic Heart Disease Using Machine Learning

## Introduction
Chronic Heart Disease (CHD) remains one of the leading causes of mortality worldwide. Early prediction of heart disease can help in taking preventive measures, improving patient outcomes, and reducing healthcare costs. This project aims to predict the likelihood of a person having heart disease using a Random Forest model trained on data from the **National Health and Nutrition Examination Survey (NHANES)**.

The NHANES dataset is a comprehensive survey that assesses the health and nutritional status of adults and children in the United States. The dataset includes medical, demographic, and lifestyle factors that contribute to chronic diseases, including CHD. More information about NHANES can be found [here](https://www.cdc.gov/nchs/nhanes/index.htm).

## Dataset Overview
The data used in this project has been collected from NHANES cycles spanning from **2005 to 2020**. Relevant features were selected based on their impact on heart disease and risk factors, as indicated in medical research. The following features were identified as the most important based on the trained Random Forest model:

### Selected Features
- Age
- LDL
- Fasting Glucose
- BMI
- HDL
- Systolic BP
- Hypertension
- Sex
- Diabetes
- Smoking

## Data Preprocessing Steps
To create the final dataset, the following preprocessing steps were performed:
1. **Feature Selection** - Identified key risk factors and features affecting heart disease based on medical literature.
2. **Data Collection** - Extracted relevant data from multiple NHANES cycles (2005-2020).
3. **Data Integration** - Merged data from different cycles using the **Sequential Number** (unique for each cycle).
4. **Data Cleaning** - Concatenated all cycles, handled missing values by dropping null records
5. **Renamed features** to more readable names from NHANES technical documentation.
6. **Standarization**
7. **Modeling**

## Machine Learning Model
The best-performing model for this classification task was **Random Forest**, optimized using **Grid Search**. The final model was trained with the best hyperparameters:
- **n_estimators** = 50
- **max_depth** = 20

### Model Performance
The model achieved outstanding performance, as shown below:
- **Accuracy:** 96.50%
- **Precision:** 96.54%
- **Recall:** 96.50%
- **F1-score:** 96.51%

## Visualization
### Confusion Matrix

![Confusion Matrix](https://github.com/omarEssam-11/Chronic_Heart_Disease_NHANES/blob/main/src/mx.png)

### ROC-AUC Curve
The **ROC-AUC** curve provides a graphical representation of the model’s ability to distinguish between the two classes.

![ROC-AUC Curve](https://github.com/omarEssam-11/Chronic_Heart_Disease_NHANES/blob/main/src/roc-auc.png)


## Saving and Loading the Model
The trained model is saved as **`CHD_RF.pkl`** and can be loaded for future predictions:
```python
import joblib
model = joblib.load('CHD_RF.pkl')
predictions = model.predict(X_test)
```

## Conclusion
This project demonstrates how machine learning, particularly Random Forest, can be used to predict chronic heart disease effectively. With high precision and recall, this model can assist healthcare professionals in early detection and intervention for individuals at risk of CHD.

---
**Contributors:** Omar Essam  
**License:** MIT  
**Data Source:** [NHANES]([https://www.cdc.gov/nchs/nhanes/index.htm](https://www.cdc.gov/))

