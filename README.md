# Iris Dataset Classification with Logistic Regression

This project uses logistic regression to classify flowers in the Iris dataset. It includes model training, evaluation, and visualization steps, along with cross-validation for model robustness.

## Overview

1. Loads the Iris dataset and splits it into training and testing sets.
2. Trains a logistic regression model.
3. Evaluates model accuracy, classification report, and confusion matrix.
4. Uses cross-validation to assess model performance.
5. Visualizes data with PCA (Principal Component Analysis) in 2D space.
6. Shows feature importance using a heatmap of logistic regression coefficients.

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Running the Code

To execute the script, run:
```bash
python ish.py
```

### Expected Output

- **Accuracy**: Average cross-validation accuracy of around 98%.
- **Visualization**: PCA plot for decision boundaries and a heatmap of feature importance.

## Files

- `ish.py`: Main code file.
- `requirements.txt`: Python package dependencies.
- `README.md`: Project documentation.

## License

This project is licensed under the MIT License.
