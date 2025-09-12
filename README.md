---

# 🌌 Cosmic Collision: Analysing Asteroid Risks with Data

## 📖 Overview

This project leverages **data analytics** and **machine learning** to evaluate the risk posed by asteroids approaching Earth. Using NASA’s asteroid close-approach dataset, the analysis focuses on predicting whether an asteroid is potentially hazardous to Earth based on features such as:

* Size and estimated diameter
* Orbital parameters
* Velocity and relative speed
* Proximity to Earth’s orbit

The notebook includes **data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling** to assess asteroid risk.



## 📂 Repository Structure

```
├── Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb
├── dataset.csv                                                 
├── processed_dataset_after_imputation.csv                      
├── normalized_dataset.csv                                      
├── README.md                                                   
├── LICENSE                                                     
```


## 🔑 Key Objectives

1. **Data Cleaning & Preprocessing**

   * Handle missing values, normalize features, and encode categorical variables.

2. **Exploratory Data Analysis (EDA)**

   * Visualize asteroid sizes, orbital parameters, and risk indicators.
   * Identify patterns and correlations in the dataset.

3. **Feature Engineering**

   * Extract meaningful variables for risk prediction.

4. **Modeling & Risk Prediction**

   * Apply machine learning models to classify asteroids as *hazardous* or *non-hazardous*.
   * Evaluate performance with accuracy, precision, recall, and F1-score.

5. **Insights & Conclusion**

   * Highlight critical factors influencing asteroid hazard predictions.

---

## 🛠️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/Cosmic-Collision-Asteroid-Risk.git
cd Cosmic-Collision-Asteroid-Risk
```

Install dependencies (recommended: create a virtual environment first):

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb
```

---

## 📊 Dataset Information

* Source: [NASA Near-Earth Object Data](https://cneos.jpl.nasa.gov/)
* **Rows**: \~4,500 asteroid observations
* **Features**: \~24 attributes including:

  * Estimated diameter
  * Relative velocity
  * Miss distance (closest approach to Earth)
  * Orbital eccentricity
  * Absolute magnitude (H)
  * Hazardous flag (target variable)

---

## 📈 Results & Findings

* Certain **orbital features** (e.g., velocity, miss distance) play a significant role in determining asteroid risk.
* Preprocessed and normalized datasets improve model performance.
* Machine learning models can provide **early risk assessment**, helping prioritize potentially hazardous asteroids.

---

## 🚀 Future Work

* Extend dataset with additional NASA missions’ data.
* Apply **deep learning models** for improved predictions.
* Develop a **real-time dashboard** for asteroid risk monitoring.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repository
* Create a new branch (`feature-xyz`)
* Commit your changes
* Open a pull request

---

## 🙌 Acknowledgements

* [NASA CNEOS](https://cneos.jpl.nasa.gov/) for providing asteroid datasets.
* Open-source libraries: **Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn**.

---
