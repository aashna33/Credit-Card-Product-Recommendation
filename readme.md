# Credit Card Customer Segmentation Project


## Summary

This project presents a comprehensive analysis of customer segmentation for credit card products using machine learning techniques. We successfully identified four distinct customer segments based on demographic, financial behavior, and transaction patterns. Each segment represents unique customer profiles with specific needs and behaviors, enabling targeted marketing strategies and personalized product recommendations.

The four identified customer segments are:
1. **Cautious Seniors:** Older customers with conservative spending habits
2. **Balanced Mid-Lifers:** Middle-aged customers with moderate engagement
3. **Young Families:** Younger individuals with family responsibilities and strong engagement
4. **High-Spending Professionals:** Frequent, high-value transactors with selective relationships

The project utilized K-Means clustering after dimensionality reduction with Principal Component Analysis (PCA) to create meaningful customer segments. For each segment, we've developed tailored product recommendations that align with their financial behaviors and needs.

This segmentation framework provides banks with actionable insights to enhance customer experience, increase product adoption, and improve customer retention through personalized offerings.

## Project Overview

### Business Context
Financial institutions typically offer a variety of credit card products to cater to different customer needs. However, without proper customer segmentation, these offerings might not effectively target the right customers. This project aims to solve this challenge by identifying distinct customer segments based on their financial behaviors, demographics, and transaction patterns.

### Objectives
1. Identify distinct customer segments based on their financial behaviors and demographics
2. Understand the characteristics and needs of each segment
3. Develop targeted product recommendations for each segment
4. Provide a framework for personalized marketing strategies

### Methodology
The project follows a structured data science approach:
1. Data collection and understanding
2. Data preprocessing and feature engineering
3. Dimensionality reduction using PCA
4. Customer segmentation using clustering algorithms
5. Cluster interpretation and profiling
6. Product recommendation development

## Data Understanding

### Dataset Overview
The dataset contains information about credit card customers, including demographic data, relationship information, and transaction details. The original dataset appears to be from a bank's customer database, stored in an Excel file named "BankChurners.xlsx".

### Key Features
The data includes various attributes such as:
- Customer demographic information (age, gender, education level, marital status)
- Relationship information (months on book, number of products held)
- Transaction behavior (transaction count, transaction amount)
- Credit behavior (credit limit, revolving balance)
- Contact information (number of contacts in the last 12 months)

### Initial Data Exploration
The initial exploration involved examining the first few rows, checking the data shape, analyzing descriptive statistics, and identifying missing values. The data preparation included removing unnecessary columns like the Naive Bayes classifier columns and customer ID.

## Data Preprocessing

### Data Cleaning
- Removed irrelevant columns including two Naive Bayes classifier columns, client number, attrition flag, and card category
- Checked for missing values in the dataset
- Examined data types and statistical properties

### Categorical Encoding
Categorical variables were encoded using Label Encoding for the following features:
- Gender
- Dependent count
- Education level
- Marital status
- Income category

### Feature Scaling
Standardization was applied to numerical features using StandardScaler to ensure all features contribute equally to the clustering algorithm:

```python
scaler = StandardScaler()
num_cols = scaled_data.select_dtypes(include=['number']).columns.to_list()
scaled_data[num_cols] = scaler.fit_transform(scaled_data[num_cols])
```

This step ensures that variables with larger scales don't dominate the clustering process.

## Feature Engineering

Several derived features were created to capture additional insights about customer behavior:

### Activity Features
```python
# Customer activity in the last 12 months
data['Months_Active_12_mon'] = 12 - data['Months_Inactive_12_mon']
data['Contacts_per_Month'] = data['Contacts_Count_12_mon'] / (data['Months_Active_12_mon'] + 1)
```
These features help understand how active customers are and how frequently they interact with the bank.

### Card Usage Features
```python
# Monthly card usage patterns
data['card_usage_count_per_month'] = data['Total_Trans_Ct'] / (data['Months_on_book'] + 1)
data['card_usage_amount_per_month'] = data['Total_Trans_Amt'] / (data['Months_on_book'] + 1)
data['Remaining_Credit'] = data['Credit_Limit'] - data['Total_Revolving_Bal']
```
These metrics provide insights into how customers use their credit cards over time.

### Age-based Features
```python
# Age-normalized dependency ratio
data['Dependents_by_Age'] = (data['Dependent_count'] + 1) / data['Customer_Age']
```
This feature normalizes the number of dependents by customer age, providing a life-stage indicator.

### Transaction Features
```python
# Transaction intensity and frequency
data['Amount_Per_Transaction'] = data['Total_Trans_Amt'] / data['Total_Trans_Ct']
data['Trans_Amount_per_Month'] = data['Total_Trans_Amt'] / (data['Months_on_book'] + 1)
data['No_of_trans_per_Month'] = data['Total_Trans_Ct'] / (data['Months_on_book'] + 1)
```
These features help understand transaction patterns and spending behavior.

### Credit Utilization
```python
# Credit utilization percentage
data['Credit_Utilization_Percentage'] = (data['Total_Revolving_Bal'] / data['Credit_Limit']) * 100
```
This is a critical metric for understanding how customers manage their available credit.

### Relationship Features
```python
# Customer relationship metrics
data['Relationship_Intensity'] = data['Total_Relationship_Count'] / (data['Months_on_book'] + 1)
data['Contact_Rate'] = data['Contacts_Count_12_mon'] / 12
data['Spending_per_Relationship'] = data['Total_Trans_Amt'] / (data['Total_Relationship_Count'] + 1)
data['Age_to_Credit_Ratio'] = data['Customer_Age'] / data['Credit_Limit']
```
These features capture the depth and breadth of customer relationships with the bank.

## Dimensionality Reduction

Principal Component Analysis (PCA) was applied to reduce the dimensionality of the dataset while preserving the variance in the data.

```python
pca = PCA(n_components=3)
pca.fit(scaled_data)
PCA_df = pd.DataFrame(pca.transform(scaled_data), columns=(["col1", "col2", "col3"]))
```

The use of PCA with 3 components helped:
- Reduce the "curse of dimensionality"
- Improve clustering performance
- Facilitate visualization of clusters

The resulting 3-dimensional dataset (PCA_df) was used for subsequent clustering analysis.

## Clustering Analysis

Two clustering methods were evaluated to identify distinct customer segments:

### K-Means Clustering
K-Means clustering was applied to the PCA-transformed data to group customers with similar characteristics.

```python
kmeans_model = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_model.fit(PCA_df)
PCA_df['Clusters'] = kmeans_model.labels_
data['Clusters'] = kmeans_model.labels_
```

### Agglomerative Clustering
Hierarchical clustering (Agglomerative Clustering) was also evaluated as an alternative approach:

```python
agglo = AgglomerativeClustering(n_clusters=k, linkage='ward')
clusters = agglo.fit_predict(PCA_df)
```

### Determining Optimal Clusters
Several methods were used to determine the optimal number of clusters:

#### Elbow Method
The Elbow method plots the Within-Cluster Sum of Squares (WCSS) against the number of clusters to identify the "elbow point" where adding more clusters provides diminishing returns.

#### Silhouette Score
The Silhouette score measures how similar each point is to its own cluster compared to other clusters. Higher scores indicate better-defined clusters.

#### Davies-Bouldin Index
The Davies-Bouldin Index measures the average similarity between clusters. Lower values indicate better cluster separation.

Based on these methods, 4 clusters were determined to be optimal for this dataset, providing a balance between cluster separation and interpretability.

## Cluster Interpretation

The clustering analysis identified four distinct customer segments, each with unique characteristics:

### Cluster 0: "Cautious Seniors"
**Key Characteristics:**
- Age: Oldest customers (~55 years)
- Total Relationships: Mostly low (0-3), peaking at 3
- Dependents by Age: Very low (0-0.05)
- Total Transactions: Low spending, with peaks at 2000 and 5000
- Transaction Amount per Month: Low values (peaks at 50-100)
- Total Transaction Count: Peaks at 35-40 and 70-80
- Card Usage per Month: Low (1-3 times)
- Relationship Intensity: Moderate (1.1-1.15)

**Customer Profile:**
Older individuals who are financially stable but demonstrate conservative spending habits. They have fewer dependents and use their credit cards infrequently, preferring controlled transactions. They maintain moderate relationships with the bank but aren't highly engaged across multiple products.

### Cluster 1: "Balanced Mid-Lifers"
**Key Characteristics:**
- Age: Mid-range (40-55), normally distributed around 45
- Total Relationships: Peaks at 3, declining after
- Dependents by Age: Moderate (0.05-0.1)
- Total Transactions: Low-to-moderate, with peaks at 1000 and 3000
- Transaction Amount per Month: Low (peaks at 50 and 100)
- Total Transaction Count: Moderate, peaking at 40 and 65
- Card Usage per Month: Low (1-3 times)
- Relationship Intensity: Slightly lower than Cluster 0 (1-1.1)

**Customer Profile:**
Middle-aged customers with balanced financial commitments. They have moderate family responsibilities and maintain controlled spending habits. Their engagement with the bank is moderate, focusing on a few key financial products.

### Cluster 2: "Young Families"
**Key Characteristics:**
- Age: Youngest segment (35-40)
- Total Relationships: Increases beyond 3
- Dependents by Age: Highest (0.09-0.125)
- Total Transactions: Moderate, peaking at 2300 and 5000
- Transaction Amount per Month: Higher than Clusters 0 & 1 (~100)
- Total Transaction Count: Peaks at 70-80
- Card Usage per Month: Normally distributed with a peak at 3
- Relationship Intensity: Higher (1.15)

**Customer Profile:**
Younger individuals with growing financial responsibilities, likely supporting families. They actively engage with multiple financial services and use their credit cards regularly for everyday expenses. Their relationship with the bank is deeper and more diverse than other segments.

### Cluster 3: "High-Spending Professionals"
**Key Characteristics:**
- Age: Mid-range (~40), normally distributed
- Total Relationships: Peaks at 2, sharply declining after
- Dependents by Age: Moderate (0.06-0.08)
- Total Transactions: Highest of all clusters, peaking at 8000 and 15000
- Transaction Amount per Month: Highest of all (~250-400)
- Total Transaction Count: Highest (100-120)
- Card Usage per Month: Peaks at 4
- Relationship Intensity: Lowest (1.05)

**Customer Profile:**
High-earning professionals who use their credit cards frequently for large transactions. They are selective about their financial relationships, focusing on fewer products but with high engagement. Their spending patterns indicate affluence and comfort with credit-based transactions.

### Cluster Comparison Table

| Segment | Age | Spending & Transactions | Relationships | Dependents | Card Usage | Engagement |
|---------|-----|-------------------------|---------------|------------|------------|------------|
| Cautious Seniors | 55+ | Low, controlled | Moderate (peak at 3) | Very low | Rare (1-3x/month) | Moderate |
| Balanced Mid-Lifers | 40-55 | Low-to-moderate | Moderate (peak at 3) | Medium | Rare (1-3x/month) | Low |
| Young Families | 35-40 | Moderate | High (3-7) | High | Frequent (3x/month) | High |
| High-Spending Professionals | 40 | Very high | Low (1-2) | Medium | Frequent (4x/month) | Low |

## Product Recommendations

Based on the identified customer segments, we have developed tailored product recommendations:

### Cluster 0: Cautious Seniors
**Recommended Products:**
- Low-Maintenance Savings Accounts – High-interest or senior citizen savings accounts with minimal fees and easy access
- Fixed Deposits & Retirement Investment Plans – Long-term, low-risk investment options
- Minimal-Fee Credit Cards – Cards with low or no annual fees and cashback on essentials
- Personalized Wealth Management – Advisory services focusing on asset preservation and estate planning
- Medical Insurance & Health Financing – Tailored health insurance plans with senior citizen benefits

### Cluster 1: Balanced Mid-Lifers
**Recommended Products:**
- Premium Credit Cards with Cashback & Rewards – Cards with benefits on dining, shopping, and fuel expenses
- Home Loans & Mortgage Advisory – Customized mortgage solutions and refinancing options
- Family-Oriented Insurance Plans – Life insurance and health coverage for dependents
- Investment in Mutual Funds & SIPs – Systematic investment plans for long-term growth
- Salary Accounts with Perks – Accounts with zero balance requirements and lifestyle rewards

### Cluster 2: Young Families
**Recommended Products:**
- Flexible Credit Cards with Higher Limits – Cards with rewards on everyday spending categories
- Child Education & Family Investment Plans – Education savings and child insurance policies
- Auto Loans & Home Loans – Competitive rates for major life purchases
- Comprehensive Family Health Insurance – Plans covering spouse and children
- Buy Now, Pay Later (BNPL) & EMI-based Financing – Flexible payment options for essential purchases

### Cluster 3: High-Spending Professionals
**Recommended Products:**
- Premium & Luxury Credit Cards – High-limit cards with exclusive benefits and travel perks
- Wealth Management & Private Banking – Personalized advisory and investment management
- High-Yield Investment Opportunities – Portfolio management and alternative investments
- International Banking & Forex Services – Seamless global transactions and premium forex accounts
- Exclusive Business Banking & Credit Solutions – Tailored financial products for entrepreneurs

### Product Recommendations Summary Table

| Segment Name | Recommended Banking Products |
|-------------|-------------------------------|
| Cautious Seniors | Senior citizen savings accounts, fixed deposits, low-fee credit cards, retirement plans, medical insurance |
| Balanced Mid-Lifers | Premium credit cards, home loans, family insurance, SIPs & mutual funds, salary accounts with perks |
| Young Families | Flexible credit cards, education & family investment plans, auto/home loans, health insurance, EMI-based financing |
| High-Spending Professionals | Luxury credit cards, private banking & wealth management, high-return investments, forex & international banking, business financial services |

## Conclusion

This project successfully identified four distinct customer segments within the credit card customer base, each with unique characteristics and financial needs. The segmentation provides a foundation for targeted marketing strategies and personalized product offerings that can enhance customer satisfaction, increase product adoption, and improve retention.

### Key Achievements
- Developed a robust customer segmentation framework using machine learning techniques
- Identified four meaningful and actionable customer segments
- Created tailored product recommendations for each segment
- Provided a strategic implementation roadmap

## Appendix: Code Breakdown

### Library Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
```

The project uses several key libraries:
- pandas and numpy: For data manipulation and analysis
- matplotlib and seaborn: For data visualization
- sklearn: For machine learning algorithms and preprocessing
- scipy: For hierarchical clustering techniques

### Data Loading
```python
def load_data(filepath):
    if file_path[-3:] == 'csv':
        return pd.read_csv(file_path)
    elif file_path[-3:] == 'txt':
        return pd.read_csv(file_path, delimiter='\t')
    elif file_path[-4:] == 'json':
        return pd.read_json(file_path)
    elif file_path[-4:] == 'xlsx':
        return pd.read_excel(file_path)
    else:
        print("An error occured, use ['csv', 'xlsx', 'txt', 'json'] files only")

file_path = r"C:\Users\Atharva\Desktop\rxib\ML_Projects\Credit_Card_Customer_Segmentation_KMeans\BankChurners.xlsx"
data = load_data(file_path)
```

A flexible function was created to handle different file formats. The data was loaded from an Excel file named "BankChurners.xlsx".

### Data Preprocessing
Data cleanup involved removing unnecessary columns and handling categorical variables:

```python
def drop_cols(df, cols_to_remove):
    df.drop(cols_to_remove, axis=1, inplace=True)

drop_cols(data,
['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
'CLIENTNUM','Attrition_Flag', 'Card_Category'])
```

Categorical encoding:
```python
cat_cols = ['Gender', 'Dependent_count', 'Education_Level','Marital_Status', 'Income_Category']
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])
```

### Feature Engineering
Multiple derived features were created to better capture customer behavior:

```python
# Examples of feature engineering
data['Months_Active_12_mon'] = 12 - data['Months_Inactive_12_mon']
data['card_usage_count_per_month'] = data['Total_Trans_Ct'] / (data['Months_on_book']+1)
data['Credit_Utilization_Percentage'] = (data['Total_Revolving_Bal'] / data['Credit_Limit']) * 100
data['Relationship_Intensity'] = data['Total_Relationship_Count'] / (data['Months_on_book']+1)
```

These engineered features provide more meaningful variables for clustering and interpretation.

### Dimensionality Reduction
PCA was used to reduce the dataset to 3 dimensions:

```python
pca = PCA(n_components=3)
pca.fit(scaled_data)
PCA_df = pd.DataFrame(pca.transform(scaled_data), columns=(["col1","col2", "col3"]))
```

### Clustering Analysis
Different methods were evaluated to determine the optimal number of clusters:

```python
# Elbow Method
k_range = range(2,20)
wcss = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(PCA_df)
    wcss.append(kmeans.inertia_)

# Final K-Means Model
kmeans_model = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_model.fit(PCA_df)
PCA_df['Clusters'] = kmeans_model.labels_
data['Clusters'] = kmeans_model.labels_
```

### Cluster Visualization
Cluster distributions were visualized to understand the differentiating features:

```python
def plot_cluster_distributions(data, cluster_col, num_cols):
    for col in num_cols:
        plt.figure(figsize=(10,5));
        for cluster in data[cluster_col].unique():
            sns.kdeplot(data[data[cluster_col]==cluster][col], label = cluster, linewidth =2)
        plt.title("Distribution of " + col);
        plt.legend();
        plt.grid();
        plt.show();
```

These visualizations helped identify the key features that differentiate the clusters, leading to the segment profiles described in this document.
