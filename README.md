# Credit Card Customer Segmentation using Unsupervised Learning

## 1. Project Overview

### 1.1 What is this project about?
This notebook applies **unsupervised machine learning** to perform **customer segmentation** on anonymized credit card data. We explore four clustering algorithms—**K-Means**, **Ward Hierarchical**, **DBSCAN**, and **Gaussian Mixture Models (GMM)**—using **PCA** and **t-SNE** for visualization. The goal is to uncover hidden customer segments based on behavioral patterns without any predefined labels.

### 1.2 Why cluster credit-card customers?
Segmenting customers enables banks to tailor products:

- **Revolvers** (carry balances): High-interest revenue, high risk.
- **Transactors** (pay in full): Low risk, high interchange revenue.
- **Dormants**: Low to no activity—opportunity for re-engagement.

Clustering can guide credit strategies, limit adjustments, targeted promotions, and default risk management.

### 1.3 Algorithms and Learning Type

| Method                | Role                                 |
|-----------------------|--------------------------------------|
| K-Means               | Baseline partitioning method         |
| Ward Hierarchical     | Variance-based hierarchical clustering |
| DBSCAN                | Density-based, detects outliers      |
| Gaussian Mixture      | Probabilistic soft clustering        |
| PCA, t-SNE            | Dimensionality reduction for visualization |

---

## 2. Dataset Provenance & Characteristics

- **Source**: Kaggle — [Credit Card Data for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
- **APA citation**:  
  Bhasin, A. (2018). *Credit Card Data for Clustering* [Data set]. Kaggle. https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

### Dataset Overview

- **Shape**: 8,950 customers × 18 columns
- **Variable Families**: Balances, purchases, cash advances, credit limits, payments, frequencies
- **Data Type**: Fully numerical (after ID removal)

---

## 3. Data Cleaning Strategy

### 3.1 Key Issues & Fixes

| Issue                  | Description                                  | Fix                            |
|------------------------|----------------------------------------------|--------------------------------|
| Missing values         | CREDIT_LIMIT (1), MINIMUM_PAYMENTS (313)     | Median imputation              |
| CUST_ID                | Non-numeric ID                               | Dropped                        |
| Skewness & outliers    | Heavily skewed monetary values               | Yeo-Johnson transform          |
| Feature scaling        | Mixed units (dollars, frequencies)           | StandardScaler (z-score)       |
| Near-zero variance     | TENURE mostly 12                             | Retained for reference         |

### 3.2 Cleaning Pipeline

1. Drop `CUST_ID`
2. Impute missing values using median strategy
3. Yeo-Johnson power transformation to address skew
4. Z-score standardization for all features

### 3.3 Post-Clean Status

- No missing values
- All 17 features scaled, numeric
- Outliers retained for clustering value

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Distributional Insights

- **Long-tail**: BALANCE, PURCHASES, CASH_ADVANCE
- **Bimodal**: FREQUENCY metrics spike at 0 and 1
- **Binary behavior**: PRC_FULL_PAYMENT ≈ 0 or 1

### 4.2 Correlation Highlights

- PURCHASES highly correlated with its components
- BALANCE correlates with CREDIT_LIMIT and CASH_ADVANCE
- PCA shows top 2 components explain **48%** of variance

### 4.3 Advanced Stats

- **Skew > 5**, **Kurtosis > 30** on most dollar columns → justifies transformation
- **Levene's test** confirms variance differences across clusters

---

## 5. Modelling Methodology

### 5.1 Algorithm Lineup

| Model             | Parameter Search        | Notes                        |
|------------------|-------------------------|------------------------------|
| K-Means          | k = 2 to 10             | Elbow & silhouette method    |
| Ward Hierarchical| k = 5 dendrogram cut    | Nested structure             |
| GMM              | n = 5, full covariance  | Soft probabilistic clustering|
| DBSCAN           | ε via 5-NN plot         | Outlier detection            |

### 5.2 Dimensionality Strategy

- PCA with 6 components captures **88%** variance—used for visual inspection
- PCA projections and t-SNE maps aid interpretability

### 5.3 Evaluation Metrics

- Silhouette Score
- Adjusted Rand Index (ARI)
- Visual assessments (PCA, t-SNE)
- Dendrogram structure for Ward method

---

## 6. Results & In-Depth Analysis

### 6.1 Visual Summary

- **Elbow plot** justifies **k = 5**
- **Silhouette scores** support this choice (peak at k = 2, but k = 5 still acceptable)
- **PCA/t-SNE** show clear cluster separation
- **Dendrogram** from Ward method aligns with K-Means

### 6.2 Quantitative Model Comparison

| Model               | #Clusters | Silhouette | ARI vs K-Means | Insight                               |
|---------------------|-----------|------------|----------------|----------------------------------------|
| K-Means             | 5         | ~0.33      | –              | Clear, scalable baseline               |
| Ward Hierarchical   | 5         | ~0.35      | 0.35           | Structurally matches K-Means           |
| Gaussian Mixture    | 5         | ~0.34      | 0.32           | Confirms cluster overlaps              |
| DBSCAN              | 2+noise   | ~0.50      | 0.07           | Detects anomalies, not segments        |

### 6.3 Cluster Profiles (from K-Means)

| Segment               | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| **C0 – Revolvers**     | High BALANCE, low PRC_FULL_PAYMENT, many CASH_ADVANCE         |
| **C1 – Transactors**   | High PURCHASES, high PRC_FULL_PAYMENT                         |
| **C2 – Dormants**      | Near-zero activity across features                            |
| **C3 – Installment**   | High INSTALLMENTS_PURCHASES, low one-off spend                |
| **C4 – Cash-Advance**  | High CASH_ADVANCE frequency and transactions                  |

---

## 7. Discussion & Strategic Conclusion

### 7.1 Business Interpretation

| Segment            | Share | Revenue Potential            | Strategy Recommendation                            |
|--------------------|-------|------------------------------|----------------------------------------------------|
| **Revolvers**       | 11%   | High interest revenue        | Credit-risk review, balance transfer offers        |
| **Transactors**     | 4%    | High interchange             | Loyalty programs, spend incentives                 |
| **Dormants**        | 35%   | Low revenue                  | Reactivation campaigns                             |
| **Installment**     | 34%   | Moderate fees                | Upsell instalment or buy-now-pay-later programs    |
| **Cash-Advance**    | 15%   | High but unstable revenue    | Financial wellness tools, lower-APR loans          |

### 7.2 Technical Takeaways

- Skew mitigation and scaling are essential
- Clustering consistency across multiple models increases confidence
- PCA/t-SNE offer crucial stakeholder communication tools

### 7.3 Model Trade-offs

- **DBSCAN**: excels at anomaly detection but fails on dense clusters
- **GMM**: adds flexibility but loses simplicity
- **PCA-KMeans**: slight performance gain but harder to interpret

### 7.4 Future Work

- **Feature Engineering**: utilization rates, purchase velocity
- **Temporal Clustering**: behavioral trajectory analysis
- **Supervised Overlay**: link clusters to default/churn risk
- **Autoencoders**: learn latent, non-linear feature spaces

---

## 8. Appendix

### Tools Used

- **Python Libraries**: `numpy`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`
- **Algorithms**: KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture, PCA, t-SNE
- **Metrics**: Silhouette Score, Adjusted Rand Index, Inertia, Calinski-Harabasz

---

## 9. Citations

Bhasin, A. (2018). *Credit Card Data for Clustering* [Data set]. Kaggle. https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
