# AI Course Projects - CA3: Machine Learning Classification

## Overview

This repository contains the implementation for Computer Assignment 3 (CA3) of the AI course, focusing on machine learning classification algorithms for student grade prediction. The project implements and compares multiple classification algorithms on a student performance dataset.

## Project Structure

```
CA3/
├── ml.ipynb                      # Main implementation notebook
├── Grades.csv                    # Original student grades dataset
├── encoded_data.csv              # Preprocessed and encoded dataset
├── AI_CA3_810102303.rar         # Complete project archive
├── AI-S04-CA3_2.pdf             # Assignment description
└── Readme.md                    # This file
```

## Dataset Description

The project uses a student performance dataset ([`Grades.csv`](CA3/Grades.csv)) containing various features about students' academic and personal information to predict their final grades.

### Target Variable
- **finalGrade**: Categorized into 4 classes:
  - **0**: Excellent (17-20)
  - **1**: Good (14-17) 
  - **2**: Average (10-14)
  - **3**: Poor (0-10)

### Key Features
- **Demographics**: Age, gender, address type
- **Family**: Mother/father education and jobs
- **Academic**: Study time, failures, school support
- **Social**: Free time, going out, romantic relationships
- **Health**: Daily/weekend alcohol consumption
- **Other**: Travel time, internet access, absences

## Data Preprocessing

### 1. Missing Data Handling
- Verified no missing values in the dataset
- All data points retained for maximum information utilization

### 2. Feature Engineering
- **Grade Categorization**: Converted continuous grades to 4 categorical classes
- **Binary Encoding**: Converted yes/no features to 1/0
- **One-Hot Encoding**: Applied to categorical variables
- **Standardization**: Used Z-score standardization for feature scaling

### 3. Train-Test Split
- 80% training, 20% testing

## Implemented Algorithms

### 1. Naive Bayes
- **Implementation**: [`GaussianNB`](CA3/ml.ipynb) from scikit-learn
- **Preprocessing**: Standardized features
- **Class Imbalance**: Attempted upsampling (resulted in decreased performance)
- **Performance**: Struggles with minority classes (0, 1)

### 2. Decision Tree
- **Implementation**: [`DecisionTreeClassifier`](CA3/ml.ipynb) with class balancing
- **Key Parameters**: 
  - `class_weight='balanced'` for handling imbalanced data
  - `max_depth=3` for optimal performance (93% accuracy)
- **Feature Importance**: DSGrade identified as most predictive feature
- **Overfitting Check**: Train accuracy (89%) vs Test accuracy (93%) - no overfitting detected

### 3. Random Forest
- **Implementation**: [`RandomForestClassifier`](CA3/ml.ipynb) with hyperparameter tuning
- **Optimization**: [`RandomizedSearchCV`](CA3/ml.ipynb) for parameter selection
- **Parameters Tuned**:
  - `n_estimators`: 50-300
  - `max_depth`: [3, 5, 7, None]
  - `min_samples_split`: 2-11
  - `min_samples_leaf`: 1-5

### 4. XGBoost
- **Implementation**: [`XGBClassifier`](CA3/ml.ipynb) for multi-class classification
- **Data Modification**: Removed EPSGrade_cat to prevent data leakage (was achieving 100% accuracy)
- **Optimization**: [`GridSearchCV`](CA3/ml.ipynb) for comprehensive parameter tuning
- **Parameters Tuned**:
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2, 0.3]
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [3, 5, 6, 7]
  - And more...

### 5. Decision Tree from Scratch
- **Custom Implementation**: Complete decision tree algorithm built from ground up
- **Key Components**:
  - **Entropy Calculation**: Information gain-based splitting
  - **Best Split Finding**: Evaluates all features and thresholds
  - **Recursive Tree Building**: Depth-limited tree construction
  - **Majority Vote**: Leaf node prediction

#### Custom Decision Tree Features
```python
class DecisionTree:
    def entropy(self, y)          # Calculate entropy for splitting
    def best_split(self, X, y)    # Find optimal feature and threshold
    def majority_vote(self, y)    # Determine leaf node prediction
    def build_tree(self, X, y, depth)  # Recursive tree construction
    def predict_one(self, x, node)     # Single sample prediction
```

## Key Findings

### Algorithm Performance Comparison
1. **Decision Tree**: Best overall performance (93% accuracy with max_depth=3)
2. **XGBoost**: High performance after removing data leakage
3. **Random Forest**: Good ensemble performance with parameter tuning
4. **Naive Bayes**: Struggled with class imbalance
5. **Custom Decision Tree**: Competitive performance demonstrating algorithmic understanding

### Feature Importance
- **Most Predictive**: DSGrade (Data Structures grade)
- **Secondary Features**: Various academic and personal factors
- **Insight**: Previous academic performance is the strongest predictor

### Class Imbalance Challenges
- **Problem**: Uneven distribution across grade categories
- **Attempted Solution**: Upsampling minority classes
- **Result**: Decreased performance due to synthetic data quality
- **Better Approach**: Using `class_weight='balanced'` parameter

## Results and Insights

### Model Performance
- **Best Model**: Decision Tree with 93% test accuracy
- **Feature Engineering Impact**: Proper encoding and scaling improved performance
- **Hyperparameter Tuning**: Significant improvement in Random Forest and XGBoost
- **Custom Implementation**: Successfully replicated scikit-learn decision tree functionality

### Educational Insights
- Previous academic performance (DSGrade) is the strongest predictor
- Class imbalance handling is crucial for fair evaluation
- Ensemble methods provide robustness but may not always outperform simpler models
- Data leakage prevention is critical for realistic performance assessment

## Files Description
- [`ml.ipynb`](CA3/ml.ipynb): Complete implementation with detailed explanations
- [`Grades.csv`](CA3/Grades.csv): Original student performance dataset
- [`encoded_data.csv`](CA3/encoded_data.csv): Preprocessed dataset ready for modeling
- [`AI_CA3_810102303.rar`](CA3/AI_CA3_810102303.rar): Complete project submission
