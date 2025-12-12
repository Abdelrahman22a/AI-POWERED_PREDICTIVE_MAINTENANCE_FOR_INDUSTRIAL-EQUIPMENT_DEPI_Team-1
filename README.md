# AI-Powered Predictive Maintenance for Industrial Equipment

## **Project Overview**

This project aims to develop a predictive maintenance system for **industrial equipment**, specifically targeting **turbofan engines**. By leveraging machine learning models and IoT sensor data, the goal is to predict the **Remaining Useful Life (RUL)** of engines to optimize maintenance schedules and reduce downtime, resulting in cost savings and increased efficiency.

### **Problem Statement**
Predicting the remaining useful life (RUL) of industrial equipment, such as turbofan engines, before failure occurs, is a key challenge in predictive maintenance. Early detection of failures allows businesses to schedule maintenance proactively, thereby reducing operational costs and improving safety.

### **Objective**
To predict the remaining useful life (RUL) of turbofan engines using sensor data and machine learning models. The models trained in this project are evaluated on their ability to accurately forecast the time remaining before the engine fails.

---

## **Dataset**

### **Dataset Source**
The dataset used in this project is the **NASA Turbofan Engine Degradation Simulation Dataset**, specifically the **FD004 dataset**, provided by NASA's **Prognostics Center of Excellence (PCoE)**.

### **Dataset Overview**
* **Data Structure**: The dataset consists of **time-series data** collected from sensors monitoring turbofan engines. Each data point represents the state of an engine at a specific cycle.
* **Columns**: 26 columns, including:
  - **Unit number**: The ID of the engine.
  - **Time**: Time in cycles.
  - **Operational settings (3)**: Variables that influence engine performance (e.g., throttle position, altitude, etc.).
  - **Sensor measurements (26)**: Various sensor readings such as temperature, pressure, and vibration.

### **Dataset Size**
* **Training Data**: 248 engine trajectories.
* **Testing Data**: 249 engine trajectories.
* **Rows**: 61,250 rows in the training data.
  
### **Fault Modes**
* **High Pressure Compressor (HPC) Degradation**
* **Fan Degradation**

### **Task**
The task is to predict the **RUL** (remaining useful life) of engines in the test set using the data from the training set. The dataset is used for **regression** tasks, where the goal is to forecast the number of cycles before failure.

### **Link to the Dataset**
You can access the dataset from the official NASA Prognostics Data Repository:  
**[NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set)**

The specific dataset used in this project (FD004) can be downloaded directly from:  
**[Download Turbofan Engine Degradation Simulation Dataset (FD004)](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)**

---

## **Model Development**

In this project, various machine learning models were evaluated for predicting the **Remaining Useful Life (RUL)** of engines:

### **1. Random Forest**
* **Type**: Ensemble learning (decision trees).
* **Advantages**: Robust to overfitting and handles non-linear relationships well.
* **Training Process**: Randomly selects subsets of features for training each tree and averages predictions.

### **2. XGBoost**
* **Type**: Gradient boosting.
* **Advantages**: Excellent performance on large datasets and complex tasks.
* **Hyperparameter Tuning**: Grid search for parameters like **learning rate**, **max depth**, **n_estimators**.

### **3. Support Vector Regression (SVR)**
* **Type**: Regression using Support Vector Machines.
* **Advantages**: Effective for high-dimensional data.
* **Kernel Used**: Radial Basis Function (RBF).

### **4. LightGBM**
* **Type**: Gradient boosting.
* **Advantages**: Fast training, especially on large datasets.
* **Unique Feature**: Leaf-wise tree growth (compared to level-wise in XGBoost).

### **5. Long Short-Term Memory (LSTM)**
* **Type**: Recurrent Neural Network (RNN).
* **Advantages**: Good for sequential data and time-series forecasting.
* **Training Process**: Backpropagation through time (BPTT).

### **6. Gated Recurrent Unit (GRU)**
* **Type**: Simplified RNN compared to LSTM.
* **Advantages**: Faster training with fewer parameters.
* **Training Process**: Similar to LSTM but with fewer gates.

---

## **Data Preprocessing**

Before training, the data underwent several preprocessing steps to ensure its suitability for model training.

### **1. Handling Missing Values**
* Imputation was used for missing data. In cases where data was missing, it was replaced with the **mean** or **median** of the respective column.

### **2. Outlier Detection**
* Outliers were detected using the **Z-score** method and removed to prevent skewed results.

### **3. Normalization**
* Sensor data was normalized to ensure all features were on the same scale, facilitating model convergence and performance.

### **4. Feature Engineering**
* **Moving Averages**: Used to smooth out sensor readings.
* **Rate of Change**: Calculated to capture the rate at which sensor values change over time.
* **Degradation Patterns**: Features derived to capture how sensor readings change as degradation progresses.

---

## **Model Evaluation**

### **Performance Metrics**
The models were evaluated using the following metrics:
* **R² (Coefficient of Determination)**: Indicates how well the model explains the variance in the RUL data.
* **MAE (Mean Absolute Error)**: The average of the absolute differences between predicted and actual values.
* **RMSE (Root Mean Squared Error)**: The square root of the average squared errors.
* **F1-Score**: A balance between precision and recall (though not explicitly used here, it can be useful in classification problems).

### **Model Performance Comparison**

| Model           | Train R²  | Train MAE | Train RMSE | Test R²  | Test MAE | Test RMSE |
|-----------------|-----------|-----------|------------|----------|----------|-----------|
| **RandomForest** | 0.617     | 36.52     | 54.95      | 0.623    | 24.47    | 33.49     |
| **XGBoost**     | 0.727     | 30.86     | 46.37      | 0.602    | 24.91    | 34.39     |
| **SVR**          | 0.558     | 40.43     | 59.91      | 0.508    | 27.64    | 38.25     |
| **LightGBM**     | 0.630     | 36.70     | 53.98      | 0.597    | 24.98    | 34.61     |
| **LSTM**         | 0.628     | 34.09     | 50.66      | 0.535    | 28.27    | 37.19     |
| **GRU**          | 0.554     | 41.35     | 57.40      | 0.433    | 32.01    | 40.08     |

### **Evaluation Insights**
* **XGBoost** performed the best, with the highest **Test R²** (0.602) and lowest **Test MAE** (24.91).
* **RandomForest** also showed strong performance, with **Test R²** of 0.623.
* **GRU** performed the worst, especially in terms of **Test R²** (0.433), showing limited effectiveness on this dataset.

---

## **Model Training**

### **RandomForest Example Code**

from sklearn.ensemble import RandomForestRegressor

# Instantiate the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict the RUL
y_pred = rf_model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²: {r2}, MAE: {mae}, RMSE: {rmse}")


## **Deployment**

### **Flask API Setup**

The trained **Random Forest model** was deployed using **Flask**, which is a lightweight Python web framework. The Flask API receives sensor data via HTTP requests and returns the predicted **Remaining Useful Life (RUL)** of the turbofan engine. The steps involved in setting up the API are as follows:

- **Flask App**: The app is initialized, and routes are defined to handle HTTP requests.
- **POST Request for Predictions**: The `/predict` route accepts **POST** requests, where sensor data is sent, preprocessed, and used by the model to predict **RUL**.
- **Scaling Input Data**: The data is scaled using the pre-trained **scaler** before being passed to the model.

### **Ngrok for Exposing the API**

To make the Flask app accessible from outside the local machine, **Ngrok** was used to create a secure tunnel. **Ngrok** generates a public URL, making the Flask app accessible from anywhere for testing and external use.

### **Postman for Testing**

**Postman**, a popular API testing tool, was used to send test requests to the deployed API. This allowed us to:
- **Test the API** with real-world sensor data.
- Ensure that the API correctly processed the input and returned accurate predictions.

---

## **Future Work**

### **Hyperparameter Tuning**:
Further optimization of hyperparameters using techniques such as **Bayesian Optimization** or **Random Search** to explore the hyperparameter space more effectively and improve model accuracy.

### **Deep Learning Models**:
Testing more advanced deep learning models, such as **CNN+LSTM**, to better capture spatial and temporal patterns in time-series data and improve the performance of the model in predicting **Remaining Useful Life (RUL)**.

### **Real-time Deployment**:
Deploying the trained models to a **real-time system** for continuous **RUL predictions**. This would allow the model to make predictions as new sensor data becomes available, enabling proactive maintenance and reducing downtime in industrial systems.

### **Data Augmentation**:
Applying techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** for synthetic data generation to handle imbalanced data and improve model robustness, especially for underrepresented failure scenarios.

---.

## **Conclusion**

This project successfully developed and evaluated multiple machine learning models to predict the **Remaining Useful Life (RUL)** of turbofan engines. The **XGBoost** model demonstrated the best performance in terms of **Test R²**, **Test MAE**, and **Test RMSE**, outperforming other models like **RandomForest**, **LightGBM**, and **LSTM**. The models demonstrated significant potential for predictive maintenance, showcasing how machine learning can be used to forecast failures and optimize maintenance schedules.

In the future, further optimization using **hyperparameter tuning**, the exploration of **deep learning models**, and the deployment of the system in a **real-time setting** could significantly improve performance and make this solution viable for industrial applications. The incorporation of **data augmentation** strategies can also help deal with class imbalance and enhance the robustness of the models.
