# MNIST Digit Classification and PCA

A machine learning project exploring handwritten digit classification using Scikit-learn. The project investigates binary classification of digits 7 and 9, compares multiple classification algorithms, evaluates model performance, applies hyperparameter tuning, and explores dimensionality reduction using Principal Component Analysis (PCA).

## Project Overview

This project was completed as part of the MSc Computer Science programme at the University of Jos, under CSC 810.

The experiments use the MNIST handwritten digit dataset, consisting of 70,000 grayscale images represented as 784 numerical pixel features.

The project is divided into two major tasks:

1. Binary classification: distinguishing handwritten digits 7 and 9.
2. Multiclass classification: classifying all ten digits using One-vs-Rest LinearSVC, with and without PCA.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- Jupyter Notebook

## Part 1: Binary Classification (7 vs 9)

The first experiment focuses on distinguishing handwritten 7s from handwritten 9s.

The dataset follows the conventional MNIST training and test partition. Only the relevant digit classes were selected for this experiment.

Four classification algorithms were implemented:

- Decision Tree Classifier using entropy.
- Multinomial Naive Bayes.
- Logistic Regression.
- Random Forest Classifier.

### Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve and ROC-AUC
- Precision-Recall Curve and Average Precision

### Experimental Results

| Model | Accuracy | ROC-AUC | Average Precision |
|---|---:|---:|---:|
| Decision Tree | 98% | 0.9784 | 0.9677 |
| Multinomial Naive Bayes | 93% | 0.9808 | 0.9778 |
| Logistic Regression | 97% | 0.9896 | 0.9863 |
| Random Forest | 99% | 0.9996 | 0.9996 |

Random Forest achieved the strongest overall baseline performance in the binary classification experiment.

Its confusion matrix recorded 1,015 correctly classified digit 7 samples and 1,002 correctly classified digit 9 samples, with 20 misclassifications.

## Part 2: Random Forest Hyperparameter Tuning

GridSearchCV was used to investigate Random Forest configurations.

The experiment used:

- StratifiedKFold with 3 splits.
- Accuracy as the scoring metric.
- A reproducible training subset of up to 10,000 samples.
- Parallel processing through `n_jobs=-1`.

### Best Parameters

| Parameter | Best Value |
|---|---|
| n_estimators | 200 |
| max_depth | 20 |
| min_samples_leaf | 1 |

### Results

- Best cross-validation accuracy: 98.50%
- Tuned model test accuracy: 99.12%
- Grid-search execution time: 108.3 seconds

## Part 3: Multiclass Classification and PCA

The second major experiment extends classification to all ten MNIST digit classes.

A One-vs-Rest Linear Support Vector Classifier (LinearSVC) was implemented.

To keep computation practical, the experiment used reproducible subsets of 12,000 training samples and 2,500 test samples.

Two configurations were compared:

1. LinearSVC using the original 784 features.
2. LinearSVC using 100 principal components obtained through PCA.

### Results

| Configuration | Test Accuracy | Reported Execution Time |
|---|---:|---:|
| Without PCA (784 features) | 89.64% | 4.55 seconds |
| With PCA (100 components) | 91.12% | 2.68 seconds |

The 100 retained principal components explained 91.4948% of the cumulative variance.

PCA was fitted on the training data only, and the resulting transformation was applied to both training and test data.

Note: The reported execution times cover different operations. The PCA experiment includes transformation time and classifier training, while the baseline timing measures classifier training. They should not be interpreted as a direct benchmark of computational speed.

## Key Observations

- Random Forest achieved the highest baseline accuracy in the binary classification task.
- Logistic Regression produced strong binary classification performance.
- Multinomial Naive Bayes had lower accuracy than the other binary classifiers.
- Hyperparameter tuning produced a Random Forest test accuracy of 99.12%.
- PCA reduced the feature space from 784 to 100 components.
- In the multiclass experiment, the PCA-based configuration achieved higher test accuracy than the baseline configuration.

## How to Run

1. Clone or download this repository.
2. Open `MNIST_Classification.ipynb` in Google Colab or Jupyter Notebook.
3. Install the required dependencies:

```python
!pip install numpy pandas matplotlib scikit-learn liac-arff
```

4. Run the notebook cells sequentially.

The notebook retrieves the MNIST dataset through Scikit-learn's `fetch_openml`.

An internet connection is required for the initial dataset download.

## Project Files

- `MNIST_Classification.ipynb` — Complete notebook containing implementation, experiments, evaluation outputs, and visualizations.
- `README.md` — Project documentation.

## Author

**Gyang Shedrach Emmanuel**

MSc Computer Science  
University of Jos, Nigeria

GitHub: https://github.com/GyangShady

LinkedIn: https://www.linkedin.com/in/gyang-shedrach-70a37226a/
