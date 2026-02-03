# 🛡️ Unsupervised Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

**Unsupervised Fraud Detection System** is a high-performance anomaly detection engine designed to identify fraudulent credit card transactions without the need for labeled historical data. By utilizing a multi-algorithm ensemble, the system captures complex "zero-day" fraud patterns that traditional supervised models might miss.

---

## 💡 Business Value Generated

This tool provides a critical layer of security for financial institutions by offering:

*   **⚡ Zero-Day Fraud Detection:** Unlike supervised models, this system doesn't need to "know" what past fraud looks like. It identifies anomalies based on deviations from normal behavior, catching new and evolving criminal tactics.
*   **🔍 Optimized Investigation:** Drastically reduces the "noise" of raw transaction data, flagging the top 1% most suspicious activities for human review.
*   **🛡️ Robustness to Outliers:** Implements `RobustScaler` to ensure that extreme transaction values (common in financial data) do not distort the model's accuracy.
*   **📊 Transparent Reporting:** Generates an automated **Dark Mode Executive Report**, allowing security teams to visualize anomalies in a 3D PCA space and audit model performance instantly.

---

## 🚀 Key Features

### 1. Multi-Algorithm Intelligence (Ensemble)
The system uses a "Wisdom of the Crowds" approach, combining three distinct mathematical perspectives:
*   **Isolation Forest:** Isolates anomalies by measuring how easy they are to separate from the rest of the data.
*   **PCA Reconstruction Error:** Identifies transactions that break the global mathematical structure of "normal" behavior (Highly optimized for speed, replacing slow distance-based methods).
*   **Elliptic Envelope:** Fits a robust covariance ellipsoid to detect outliers in the main data distribution.

### 2. High-Performance Processing
*   **Linear Scalability:** Designed to handle **285,000+ transactions** in seconds by utilizing optimized PCA reconstruction techniques instead of computationally expensive algorithms like LOF.
*   **Smart Preprocessing:** Uses `RobustScaler` to handle the heavy skewness and outliers present in financial transaction amounts.

### 3. Advanced Visualization & UX
*   **3D PCA Projection:** Reduces 30-dimensional transaction data into a 3D space to show where fraud clusters.
*   **Dark Mode HTML Report:** A professional, security-focused dashboard featuring performance metrics (Recall, Precision, F1-Score) and amount distribution analysis.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.7+ |
| **Machine Learning** | Scikit-Learn (Isolation Forest, PCA, Elliptic Envelope) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Mplot3d |
| **Reporting** | HTML5, CSS3 (Modern Dark Mode UI) |
