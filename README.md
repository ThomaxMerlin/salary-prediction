# salary-predict

# Salary Prediction using Linear Regression

This project demonstrates how to build a linear regression model to predict salary based on features like `YearsExperience` and `Age`. The dataset used is `Salary_Data.csv`, which contains 30 entries with numerical features.

---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Running the Code](#running-the-code)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [License](#license)

---

## **Prerequisites**
Before running the code, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install numpy pandas seaborn scikit-learn
  ```
- Jupyter Notebook (optional, for running `.ipynb` files).

---

## **Getting Started**
1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/salary-prediction.git
   cd salary-prediction
   ```

2. **Download the Dataset**  
   Ensure the dataset `Salary_Data.csv` is in the same directory as the script or notebook.

---

## **Running the Code**
1. **Using Jupyter Notebook**  
   - Open the `.ipynb` file in Jupyter Notebook.
   - Run each cell sequentially to execute the code.

2. **Using Python Script**  
   - Save the code in a `.py` file (e.g., `salary_prediction.py`).
   - Run the script using:
     ```bash
     python salary_prediction.py
     ```

---

## **Code Explanation**
### **1. Import Libraries**
```python
import numpy as np
import pandas as pd
import seaborn as sns
```
- Libraries used for data manipulation and visualization.

### **2. Load and Explore Data**
```python
data_set = pd.read_csv("Salary_Data.csv")
data_set.head()
data_set.describe()
data_set.info()
```
- Load the dataset and explore its structure, summary statistics, and data types.

### **3. Data Visualization**
```python
sns.scatterplot(data=data_set, x='YearsExperience', y='Salary')
sns.distplot(data_set['YearsExperience'])
sns.boxplot(x=data_set['YearsExperience'])
sns.displot(x=data_set['YearsExperience'])
sns.distplot(data_set['Salary'])
sns.boxplot(x=data_set['Salary'])
```
- Visualize the distribution and relationships between features using scatter plots, histograms, and box plots.

### **4. Correlation Analysis**
```python
corr_data = data_set.corr()
sns.heatmap(corr_data, annot=True, cmap='Accent')
```
- Analyze correlations between features using a heatmap.

### **5. Prepare Data for Modeling**
```python
x = data_set.drop(columns='Salary')
y = data_set.drop(columns='YearsExperience')
```
- Separate the features (`x`) and target variable (`y`).

### **6. Train-Test Split**
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
- Split the data into training and testing sets.

### **7. Build and Train Model**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
```
- Train a linear regression model on the training data.

### **8. Evaluate Model**
```python
y_predict = model.predict(x_test)
mean_squared_error(y_test, y_predict)
r2_score(y_test, y_predict)
```
- Evaluate the model using Mean Squared Error (MSE) and R² score.

### **9. Make Predictions**
```python
model.predict([[2, 4]])
```
- Predict salary for new input values.

---

## **Results**
- **Mean Squared Error (MSE)**: A measure of the model's prediction error.
- **R² Score**: Indicates how well the model explains the variance in the target variable.
- **Predictions**: The model can predict salary based on `YearsExperience` and `Age`.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as needed.

---

## **Support**
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at [minthukywe2020@gmail.com](mailto:minthukywe2020@gmail.com).

---

Enjoy exploring the salary prediction model! 🚀
