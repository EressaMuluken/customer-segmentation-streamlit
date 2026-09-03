# End-to-End Customer Segmentation with K-Means & GMM

>  **Work in Progress**
>
> This project is actively being developed. The current version uses a predefined customer dataset and provides an interactive Streamlit application for customer segmentation using K-Means and Gaussian Mixture Models (GMM). Additional functionality will be added as development continues.

## Overview

Customer segmentation is an unsupervised machine learning technique used to identify groups of customers with similar characteristics and behaviors.

The goal of this project is to explore different clustering approaches and develop an interactive application that allows users to configure, train, evaluate, and visualize customer segmentation models.

The project combines **exploratory data analysis and machine learning experimentation in a Jupyter Notebook** with an **interactive Streamlit application**.

## Project Structure

```text
customer-segmentation-streamlit/
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   └── customer_segmentation_analysis.ipynb
│
├── main.py
├── requirements.txt
└── README.md
```

### Folder and File Description

* **`data/`** — Contains the dataset used for the project.
* **`models/`** — Stores trained machine learning models and related model artifacts.
* **`notebooks/`** — Contains the exploratory analysis, preprocessing, model experimentation, and clustering analysis.
* **`main.py`** — Streamlit application providing the interactive customer segmentation interface.
* **`requirements.txt`** — Python dependencies required to run the project.
* **`README.md`** — Project documentation.

## Methodology

The project follows a typical machine learning workflow:

```text
Data
  ↓
Exploratory Data Analysis
  ↓
Data Cleaning & Preprocessing
  ↓
Feature Selection
  ↓
Feature Scaling
  ↓
Clustering
  ↓
Model Evaluation
  ↓
PCA Visualization
  ↓
Interactive Streamlit Application
```

### 1. Exploratory Data Analysis

The initial analysis is performed in the Jupyter Notebook to understand the structure and characteristics of the customer data.

This includes:

* Examining the dataset structure
* Identifying missing values
* Exploring numerical and categorical variables
* Investigating customer behavior and spending patterns
* Selecting relevant features for segmentation

### 2. Data Preprocessing

The selected features are prepared for clustering through appropriate preprocessing and feature scaling.

Scaling is particularly important for distance-based algorithms such as K-Means, where features with larger numerical ranges can otherwise dominate the clustering process.

### 3. K-Means Clustering

K-Means partitions customers into a predefined number of clusters by assigning observations to the nearest cluster centroid.

The application allows configuration of parameters such as:

* Number of clusters
* Initialization method
* Random state

`k-means++` is used as the default initialization strategy.

### 4. Gaussian Mixture Model

Gaussian Mixture Models provide a probabilistic approach to clustering. Instead of assigning observations solely to a single centroid, GMM models the data as a mixture of Gaussian distributions.

The application allows configuration of:

* Number of components
* Covariance type
* Random state

### 5. Model Evaluation

Several clustering metrics are used to evaluate the quality of the resulting segments.

| Metric                  | Preferred Direction | Purpose                                       |
| ----------------------- | ------------------- | --------------------------------------------- |
| Silhouette Score        | Higher              | Measures cluster cohesion and separation      |
| Davies-Bouldin Index    | Lower               | Measures similarity between clusters          |
| Calinski-Harabasz Score | Higher              | Measures separation between clusters          |
| Inertia                 | Lower               | Measures within-cluster variation for K-Means |
| AIC                     | Lower               | Model selection criterion for GMM             |
| BIC                     | Lower               | Model selection criterion for GMM             |

The metrics are considered together rather than relying on a single evaluation measure.

### 6. PCA Visualization

Principal Component Analysis (PCA) is used to project the processed feature space into two dimensions for visualization.

The PCA representation is used to visualize the resulting customer segments and compare clustering solutions.

> PCA is used for visualization and does not replace the original feature space used to train the clustering models.

## Streamlit Application

The Streamlit application provides an interactive interface for experimenting with the clustering models.

Users can:

* Select K-Means, GMM, or both
* Configure model parameters
* Run the selected clustering algorithms
* View clustering evaluation metrics
* Visualize customer segments using PCA
* Compare the resulting clustering solutions

The application is currently based on the predefined dataset included with the project.

## Results

The initial experiments demonstrate how K-Means and GMM can produce different customer segmentations depending on the model configuration and number of clusters.

The notebook contains the detailed exploratory analysis and model experimentation, while the Streamlit application provides an interactive way to reproduce and explore the segmentation results.

Example evaluation metrics from one K-Means configuration:

| Metric                  |   Value |
| ----------------------- | ------: |
| Silhouette Score        |   0.195 |
| Davies-Bouldin Index    |   1.748 |
| Calinski-Harabasz Score |  664.44 |
| Inertia                 | 8159.36 |

These values are specific to the configuration and dataset used in the experiment and should not be interpreted as universal benchmarks.

## Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **Joblib**
* **Jupyter Notebook**

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd customer-segmentation-streamlit
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application with:

```bash
streamlit run main.py
```

The application will open in your browser.

## Current Status

🚧 **Work in Progress**

The core clustering workflow is currently implemented, including:

* Data preprocessing
* Feature selection
* K-Means clustering
* Gaussian Mixture Models
* Cluster evaluation
* PCA visualization
* Interactive Streamlit controls

The project is still being developed and the application architecture and functionality may change.

## Future Improvements

Planned improvements include:

* [ ] Additional application features
* [ ] Support for uploading custom datasets
* [ ] Automatic feature-type detection
* [ ] More flexible preprocessing options
* [ ] Additional clustering algorithms
* [ ] Improved cluster profiling
* [ ] Automated selection of the number of clusters
* [ ] More detailed customer segment interpretation
* [ ] Improved visualization and dashboard design
* [ ] Model export and reuse

