# Customer Segmentation Using Clustering

Customer segmentation using KMeans and Gaussian Mixture Models, with model
evaluation, cluster profiling, and deployment-ready pipelines.

> **A complete ML project, from raw data → analysis → clustering → model
> evaluation → business interpretation → deployment.**

## Overview

This project explores customer segmentation using unsupervised machine
learning techniques.

The analysis uses the Customer Segmentation : Clustering dataset from Kaggle
and applies two clustering approaches:

- KMeans
- Gaussian Mixture Model (GMM)

The goal is to identify meaningful customer segments based on demographic,
spending, purchasing, and engagement characteristics.

## Methods

The analysis includes:

- Exploratory data analysis
- Data preprocessing
- Feature engineering
- Feature scaling
- KMeans clustering
- Elbow method
- Silhouette analysis
- Principal Component Analysis (PCA)
- Gaussian Mixture Models
- AIC and BIC model evaluation
- Cluster profiling
- Customer segment interpretation
- Model comparison
- Model deployment preparation using scikit-learn pipelines

## Customer Segments

The clustering analysis produced several customer profiles, including:

- Low-Value Customers
- High-Value Store Customers
- Mid-Value Active Customers
- High-Value Omnichannel Customers
- High-Potential Underengaged Customers

## Deployment

The final clustering models were prepared using scikit-learn pipelines and
can be integrated into applications such as Streamlit or an API.

## Dataset

Customer Segmentation : Clustering — Kaggle

## Project Structure

```text
customer-segmentation/
├── notebooks/
├── models/
├── README.md
├── requirements.txt
└── .gitignore