# MNIST Handwritten Digit Classification and PCA

A machine learning project exploring binary and multiclass handwritten-digit classification using the MNIST dataset and Scikit-learn.

## Project Overview

This project investigates the application and evaluation of different machine learning classification algorithms using the MNIST handwritten-digit dataset.

The experiments begin with binary classification, distinguishing between handwritten digits **7 and 9**, before extending to multiclass classification across all ten digit classes (0–9).

The project also explores hyperparameter optimisation using GridSearchCV and dimensionality reduction using Principal Component Analysis (PCA).

## Objectives

- Prepare and explore the MNIST dataset.
- Implement and compare multiple classification algorithms.
- Evaluate binary classification models using different performance metrics.
- Optimise Random Forest hyperparameters using GridSearchCV.
- Implement multiclass classification using One-vs-Rest LinearSVC.
- Investigate the effect of PCA on classification performance and computational efficiency.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset

The project uses the MNIST handwritten-digit dataset.

- 70,000 grayscale handwritten-digit images.
- Image dimensions: 28 × 28 pixels.
- 784 features per flattened image.
- Ten digit classes, ranging from 0 to 9.

The dataset is loaded using Scikit-learn's `fetch_openml` function.

## Experiments and Methodology

### 1. Binary Classification

The first experiments focus on distinguishing between digits 7 and 9.

Four classification algorithms are implemented:

- Decision Tree Classifier
- Multinomial Naive Bayes
- Logistic Regression
- Random Forest Classifier

Digit 7 is encoded as class 0, while digit 9 is encoded as class 1.

### 2. Model Evaluation

The classifiers are evaluated using:

- Classification reports
- Confusion matrices
- Precision
- Recall
- F1-score
- Receiver Operating Characteristic (ROC) curves
- Area Under the Curve (AUC)
- Precision-Recall curves
- Average Precision

These metrics provide different perspectives on classification performance.

### 3. Random Forest Hyperparameter Tuning

GridSearchCV is used to investigate different Random Forest parameter combinations.

The search considers:

- Number of estimators
- Maximum tree depth
- Minimum samples per leaf

Stratified cross-validation is used during hyperparameter selection.

For computational practicality, the grid-search experiment uses up to 10,000 training examples. The selected model is subsequently evaluated on the separate binary test set.

### 4. Multiclass Classification

The project extends classification to all ten MNIST digit classes.

An One-vs-Rest strategy is implemented with Linear Support Vector Classification (LinearSVC).

The experiment evaluates the model's training time and classification accuracy.

### 5. Principal Component Analysis (PCA)

PCA is applied to reduce the original 784-dimensional feature space to 100 principal components.

Two multiclass models are compared:

- LinearSVC trained using the original 784 features.
- LinearSVC trained using 100 PCA components.

The comparison considers training time, test accuracy and cumulative explained variance.

For computational practicality, the multiclass experiments use reproducible subsets of up to 12,000 training examples and 2,500 test examples.

## Repository Structure

```text
MNIST-Digit-Classification/
│
├── README.md
└── MNIST_Classification.ipynb
```

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open `MNIST_Classification.ipynb` in Jupyter Notebook or Google Colab.
4. Run the notebook cells sequentially.
5. Review the generated metrics, tables and visualisations.

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn liac-arff
```

The first execution requires an internet connection to download the MNIST dataset.

## Key Learning Outcomes

This project provides practical experience in:

- Data preparation and preprocessing.
- Binary and multiclass classification.
- Comparing machine learning algorithms.
- Model evaluation and interpretation.
- Hyperparameter optimisation.
- Cross-validation.
- Parallel model training.
- Dimensionality reduction.
- Understanding the relationship between model complexity and computational efficiency.

## Limitations

The GridSearchCV and multiclass experiments use reproducible subsets to keep execution practical on personal computing hardware.

Consequently, their results should be interpreted in the context of the selected sample sizes rather than assumed to represent full-dataset performance.

## Author

**Gyang Shedrach Emmanuel**

M.Sc. Computer Science  
University of Jos, Nigeria

- GitHub: https://github.com/GyangShady
- LinkedIn: https://www.linkedin.com/in/gyang-shedrach-70a37226a/
