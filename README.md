# 🌌 Cosmic Collision: Analysing Asteroid Risks with Data  

## 📖 Overview  
This project leverages **data analytics** and **machine learning** to evaluate the risk posed by asteroids approaching Earth. Using NASA’s asteroid close-approach dataset, the analysis focuses on predicting whether an asteroid is potentially hazardous to Earth based on features such as:  

- Size and estimated diameter  
- Orbital parameters  
- Velocity and relative speed  
- Proximity to Earth’s orbit  

The notebook includes **data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling** to assess asteroid risk.  

---

## 📂 Repository Structure  
├── Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb # Jupyter notebook with full analysis
├── dataset.csv # Raw asteroid close-approach dataset (~4.5k rows, 24 features)
├── processed_dataset_after_imputation.csv # Dataset after missing-value imputation
├── normalized_dataset.csv # Scaled & encoded dataset
├── README.md # Project documentation
├── LICENSE # License information (MIT)


---


---

## 🔑 Key Objectives  

### 🧹 Data Cleaning & Preprocessing  
- Handle missing values  
- Normalize features  
- Encode categorical variables  

### 📊 Exploratory Data Analysis (EDA)  
- Visualize asteroid sizes, orbital parameters, and risk indicators  
- Identify correlations and hidden patterns  

### 🛠️ Feature Engineering  
- Derive meaningful features to improve predictive accuracy  

### 🤖 Modeling & Risk Prediction  
- Train ML models to classify hazardous vs. non-hazardous asteroids  
- Evaluate with accuracy, precision, recall, and F1-score  

---

## 🛠️ Installation & Setup  

Clone the repository:  
```bash
git clone https://github.com/<your-username>/Cosmic-Collision-Asteroid-Risk.git
cd Cosmic-Collision-Asteroid-Risk

⚙️ Requirements

Python: >= 3.8

Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb

📈 Results & Insights

Identification of Potentially Hazardous Asteroids (PHAs)

Visualization of orbital parameters and risk levels

ML model performance metrics (accuracy, precision, recall, F1-score)

Key trends in asteroid size, velocity, and proximity to Earth

(You can add plots and screenshots here to showcase findings)

🌌 Future Work

Integrating real-time asteroid tracking APIs

Enhancing ML models with deep learning

Deploying an interactive dashboard for asteroid monitoring

🤝 Contributing

Contributions are welcome! 🎉

Fork the repo

Create a new branch

Submit a pull request with improvements

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

🛰️ Acknowledgments

NASA Open Data Portal

scikit-learn & Python Data Science Community
