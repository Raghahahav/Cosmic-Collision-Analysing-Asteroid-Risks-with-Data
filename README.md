# Cosmic Collision: Analyzing Asteroid Risks with Data

This project analyzes asteroid collision risks using machine learning techniques. The analysis includes exploratory data analysis, feature engineering, model training, and risk assessment of potentially hazardous asteroids.

## Project Overview

The notebook performs comprehensive analysis on asteroid data including:
- **Data Inspection & Cleaning**: Analysis of asteroid features and data quality
- **Feature Engineering**: Creation of derived features for better model performance
- **Class Imbalance Handling**: Using SMOTE for balanced training data
- **Machine Learning Models**: Random Forest and XGBoost classifiers
- **Model Evaluation**: Performance metrics, SHAP values, and feature importance
- **Anomaly Detection**: Isolation Forest for outlier detection

## Setup Instructions

### Prerequisites
- Python 3.13+ (as configured in your .venv)
- Virtual environment already created at `.venv/`

### Quick Setup
1. **Run the setup script:**
   ```bash
   # On Windows
   setup_env.bat
   
   # On Linux/Mac
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

### Manual Setup
If you prefer manual setup:

1. **Activate the virtual environment:**
   ```bash
   # On Windows
   ..\.venv\Scripts\activate.bat
   
   # On Linux/Mac
   source ../.venv/Scripts/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set up Jupyter kernel:**
   ```bash
   python -m ipykernel install --user --name cosmic-collision --display-name "Cosmic Collision Analysis"
   ```

## Running the Analysis

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   Navigate to `Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb`

3. **Select the correct kernel:**
   - In Jupyter, go to Kernel → Change Kernel
   - Select "Cosmic Collision Analysis"

4. **Run the analysis:**
   - Execute cells sequentially or run all cells

## Dataset Files

- `dataset.csv` - Original asteroid data
- `normalized_dataset (1).csv` - Normalized dataset
- `processed_dataset_after_imputation (1).csv` - Dataset after missing value treatment

## Key Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting framework
- **imbalanced-learn**: Handling imbalanced datasets
- **shap**: Model interpretability
- **seaborn/matplotlib**: Data visualization

## Project Structure

```
Cosmic collsion/
├── .venv/                          # Virtual environment
└── Cosmic-Collision-Analysing-Asteroid-Risks-with-Data/
    ├── Cosmic-Collision-Analysing-Asteroid-Risks-with-Data.ipynb
    ├── dataset.csv
    ├── normalized_dataset (1).csv
    ├── processed_dataset_after_imputation (1).csv
    ├── README.md                   # This file
    ├── requirements.txt            # Python dependencies
    ├── setup_env.bat              # Windows setup script
    └── setup_env.sh               # Linux/Mac setup script
```

## Notes

- The virtual environment is already created and configured for Python 3.13
- All dependencies will be installed within the `.venv` folder
- The Jupyter kernel "Cosmic Collision Analysis" will be available for this project
- Make sure to activate the virtual environment before running any Python scripts or notebooks

## Troubleshooting

If you encounter issues:
1. Ensure the virtual environment is activated
2. Check that all dependencies are installed: `pip list`
3. Verify the Jupyter kernel is installed: `jupyter kernelspec list`
4. Restart Jupyter if the kernel doesn't appear