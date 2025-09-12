# Cosmic-Collision-Analysing-Asteroid-Risks-with-Data
The primary objective of this analysis is to use data analytics and machine learning techniques to determine the likelihood of an asteroid being hazardous to Earth based on various features provided in the dataset. This includes examining characteristics such as the asteroid's size, orbital parameters, velocity, and proximity to Earth's orbit.
The notebook includes data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling to assess asteroid risk.


📂 Repository Structure
├── Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb   # Jupyter notebook with full analysis
├── dataset.csv                                                 # Raw asteroid close-approach dataset (~4.5k rows, 24 features)
├── processed_dataset_after_imputation.csv                      # Dataset after missing-value imputation
├── normalized_dataset.csv                                      # Scaled & encoded dataset
├── README.md                                                   # Project documentation
├── LICENSE                                                     # License information (MIT)

🔑 Key Objectives

Data Cleaning & Preprocessing

Handle missing values, normalize features, and encode categorical variables.

Exploratory Data Analysis (EDA)

Visualize asteroid sizes, orbital parameters, and risk indicators.

Identify patterns and correlations in the dataset.

Feature Engineering

Extract meaningful variables for risk prediction.

Modeling & Risk Prediction

Apply machine learning models to classify asteroids as hazardous or non-hazardous.

Evaluate performance with accuracy, precision, recall, and F1-score.

Insights & Conclusion

Highlight critical factors influencing asteroid hazard predictions.

🛠️ Installation & Setup

Clone the repository:

git clone https://github.com/<your-username>/Cosmic-Collision-Asteroid-Risk.git
cd Cosmic-Collision-Asteroid-Risk


Install dependencies (recommended: create a virtual environment first):

pip install -r requirements.txt


Open the notebook:

jupyter notebook Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb

📊 Dataset Information

Source: NASA Near-Earth Object Data

Rows: ~4,500 asteroid observations

Features: ~24 attributes including:

Estimated diameter

Relative velocity

Miss distance (closest approach to Earth)

Orbital eccentricity

Absolute magnitude (H)

Hazardous flag (target variable)

📈 Results & Findings

Certain orbital features (e.g., velocity, miss distance) play a significant role in determining asteroid risk.

Preprocessed and normalized datasets improve model performance.

Machine learning models can provide early risk assessment, helping prioritize potentially hazardous asteroids.

🚀 Future Work

Extend dataset with additional NASA missions’ data.

Apply deep learning models for improved predictions.

Develop a real-time dashboard for asteroid risk monitoring.

📜 License

This project is licensed under the MIT License. See LICENSE
 for details.

🤝 Contributing

Contributions are welcome!

Fork the repository

Create a new branch (feature-xyz)

Commit your changes

Open a pull request

🙌 Acknowledgements

NASA CNEOS
 for providing asteroid datasets.

Open-source libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

✨ "Exploring the cosmos with data, one asteroid at a time."
