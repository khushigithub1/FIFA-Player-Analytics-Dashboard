# ⚽ FIFA Player Analytics | Machine Learning & Streamlit Deployment

## Project Overview

This project aims to analyze FIFA player performance and predict a player's overall rating using Machine Learning techniques. The model analyzes player characteristics such as age, height, weight, potential, and experience to estimate the player's overall performance rating.

The project demonstrates a complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, player clustering, model building, evaluation, visualization, and deployment using Streamlit.

---


## 🎯 Objectives
* Analyze and understand the FIFA player dataset.
* Perform data cleaning and preprocessing.
* Conduct Exploratory Data Analysis (EDA).
* Engineer useful player-level features.
* Build regression models for overall-rating prediction.
* Compare model performance using evaluation metrics.
* Apply clustering techniques to identify similar player profiles.
* Create interactive visualizations and player insights.
* Deploy the final Machine Learning model using Streamlit.

---

## 🔄 Project Workflow

Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Model Building → Model Evaluation → Player Clustering → Visualization → Streamlit Deployment

---

## 📊 Dataset Information

The project uses the FIFA Players 20 dataset, containing 18,278 players and 104 original features.

The dataset includes player performance, physical, financial, club, nationality, and positional information.

Feature	Description
short_name	Player display name
age	Player age
height_cm	Player height in centimeters
weight_kg	Player weight in kilograms
overall	Current FIFA player rating (Target Variable)
potential	Potential FIFA player rating
club	Player's current club
nationality	Player nationality
value_eur	Estimated player market value
wage_eur	Player wage
player_positions	Player playing positions


---

## 🔍 Exploratory Data Analysis

The following analyses were performed:

### Univariate Analysis
* Overall Rating Distribution
* Age Distribution
* Potential Distribution
* Player Nationality Distribution
* Player Position Distribution
* Player Wage Analysis
* Player Market Value Analysis

### Bivariate Analysis
* Overall Rating vs Age
* Overall Rating vs Potential
* Player Value vs Overall Rating
* Wage vs Player Position
* Physical Attributes vs Player Performance

### Multivariate Analysis
* Correlation Analysis
* Feature Relationship Exploration
* Player Segmentation
* Dimensionality Reduction
* Feature Importance Analysis

---


## ⚙️ Feature Engineering

* A new feature was created:
* experience = age - 18
* The final prediction model uses:

age
height_cm
weight_kg
potential
experience

The target variable is:overall

---

## 🤖 Models Used
### 1. Linear Regression

* Used as the baseline regression model.

### 2. XGBoost Regressor

* Used to capture non-linear relationships between player characteristics and overall performance.

### 3. Neural Network

* Used as an additional regression approach for performance comparison.

### 4. K-Means Clustering

* Used to group players with similar characteristics and performance profiles.

---

## 📈 Model Performance
Model	R² Score
Linear Regression	0.791393
XGBoost	0.923355
Neural Network	0.921643
XGBoost Deployment Model	0.949680

## 📊 Key Findings
- XGBoost outperformed the baseline Linear Regression model.
The original notebook reported an XGBoost R² score of 0.923355.
The final deployment model achieved an R² score of 0.949680 on the clean test split.
- Player age and experience provide useful information for current performance analysis.
- Potential is strongly associated with player overall rating.
- Player clustering helps identify groups with similar performance and player characteristics.
- The deployed Streamlit application enables interactive player analysis and real-time overall-rating prediction.

---

## 🏆 Best Model

The XGBoost Regressor was selected as the final deployment model with:

* R² Score: 0.949680
* Input Features: Age, Height, Weight, Potential, Experience
* Target: Overall Rating


## 🧠 Player Clustering

* K-Means clustering was applied to group players into 5 clusters based on player characteristics and performance-related attributes.

The clustering analysis helps identify player profiles such as:

- Elite / high-performing players
- High-potential developing players
- Experienced players
- Developing players
- Lower-rated player groups

- The project also explores Hierarchical Clustering, PCA, and t-SNE for additional player segmentation and visualization.

---

## 💡 Key Insights
* Player ratings generally increase through the early and mid-20s before gradually declining.
* Higher-potential players tend to have stronger overall ratings.
* Experienced players generally show higher current performance levels.
* Player segmentation can help identify similar player profiles.
* Player value and wage provide additional information for football performance and market analysis.
* Machine Learning can support data-driven scouting and player evaluation.

---

## 💼 Business Recommendations
* Use player performance data to support scouting decisions.
* Identify young players with high potential for future development.
* Compare players using overall rating, potential, and physical attributes.
* Use clustering to identify similar player profiles.
* Use Machine Learning predictions as an additional tool for player evaluation and benchmarking.
* Combine player performance and market information for transfer analysis.

---
## 🚀 Live Demo

[![Open Live Demo](https://img.shields.io/badge/🚀%20Open%20Live%20Demo-FIFA%20Player%20Analytics-brightgreen?style=for-the-badge)](https://fifa-player-analytics-dashboard-hxg5mv5ks4sbcdhuamssdc.streamlit.app/)

---

## 🛠️ Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Plotly
* Jupyter Notebook
* Streamlit
* Git
* GitHub

---

##  🚀 Future Improvements
* Player-to-player comparison
* Advanced player filtering
* Position-specific prediction
* Interactive cluster visualization
* SHAP-based model explainability
* Transfer value prediction
* Real-time football data integration
* Automated model retraining
* Integration with newer FIFA datasets



## 📚 References
* FIFA Players 20 Dataset
* Pandas Documentation
* NumPy Documentation
* Scikit-learn Documentation
* XGBoost Documentation
* Matplotlib Documentation
* Seaborn Documentation
* Plotly Documentation
* Streamlit Documentation\


## 📬 Connect With Me
* LinkedIn: https://www.linkedin.com/in/akanksha-srivastava-20a43623b
* GitHub: https://github.com/khushigithub1


## 👩‍💻 Author
Akanksha Srivastava
Data Science & Machine Learning Enthusiast
