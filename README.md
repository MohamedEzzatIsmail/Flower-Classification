# 🌸 Flower Classification

This project is a classic machine learning problem where the goal is to classify iris flowers into one of three species — **Setosa**, **Versicolor**, or **Virginica** — based on four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model is trained using **Logistic Regression** and evaluated using accuracy score and confusion matrix.

---

## 📊 Dataset Information

The Iris dataset contains:

- 150 samples
- 3 species (target classes)
- 4 features per sample

| Feature          | Description         |
|------------------|---------------------|
| Sepal Length     | In centimeters       |
| Sepal Width      | In centimeters       |
| Petal Length     | In centimeters       |
| Petal Width      | In centimeters       |

---

## 🛠️ Technologies Used

- Python
- scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Project Structure

flower-classification/ │ ├── Flower classification.py # Main Python script ├── README.md # Project documentation

---

## 🚀 How to Run the Project

1. Clone the repository:

Install the dependencies:

pip install scikit-learn matplotlib seaborn
Run the script:
You’ll see:

The model accuracy

A classification report

A confusion matrix heatmap

A sample prediction output

🧠 Model Used
Logistic Regression from scikit-learn

Accuracy, precision, recall, and f1-score are used for evaluation

📌 Sample Prediction
Input: [5.1, 3.5, 1.4, 0.2]
Output: Predicted Iris Class: Setosa
