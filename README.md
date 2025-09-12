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

## 🔑 Key Objectives  

### 🧹 Data Cleaning & Preprocessing  
- Handle missing values  
- Normalize features  
- Encode categorical variables  

### 📊 Exploratory Data Analysis (EDA)  
- Visualize asteroid sizes, orbital parameters, and risk indicators  
- Identify patterns and correlations in the dataset  

### 🛠️ Feature Engineering  
- Extract meaningful features to improve model accuracy  

### 🤖 Modeling & Risk Prediction  
- Train ML models to classify hazardous vs. non-hazardous asteroids  
- Evaluate performance with accuracy, precision, recall, and F1-score  

---

## 🛠️ Installation & Setup  

Clone the repository:  
```bash
git clone https://github.com/<your-username>/Cosmic-Collision-Asteroid-Risk.git
cd Cosmic-Collision-Asteroid-Risk

## ⚙️ Requirements & Setup

- **Python**: >= 3.8  
- **Install dependencies**:  
  ```bash
  pip install -r requirements.txt

Open the notebook:
jupyter notebook Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb


📈 Results & Insights

Identification of Potentially Hazardous Asteroids (PHAs)

Visualization of orbital parameters and risk levels

Machine learning model performance metrics

Key trends in asteroid size, velocity, and proximity to Earth

(Add graphs/screenshots from your notebook here)

🌌 Future Work

Integrating real-time asteroid tracking APIs

Enhancing machine learning models for better predictions

Building an interactive dashboard for asteroid monitoring

🤝 Contributing

Contributions are welcome! Please fork this repo and submit a pull request with improvements.

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

🛰️ Acknowledgments

NASA Open Data Portal

scikit-learn & Python Data Science Community

