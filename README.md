# Fake News Detector using Machine Learning & NLP

This project classifies news articles as **real** or **fake** using a full machine learning pipeline with **NLP preprocessing**, **TF-IDF vectorization**, and two classification models: **Naive Bayes** and **Logistic Regression**.

Built as a final ML project using the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

---

## 📁 Project Structure

```
fake-news-detector/
├── data/             # Dataset CSVs
├── models/           # Saved .pkl model files
├── outputs/          # Confusion matrices & WordClouds
├── notebooks/        # Jupyter notebook version
├── src/              # Modular code files (preprocess, train, eval, etc.)
├── main.py           # Runs the full ML pipeline
├── requirements.txt  # Dependencies
└── README.md         # Project overview
```

---

## What the Project Does

- Preprocesses raw news article text
- Converts text to numeric features with TF-IDF
- Trains and evaluates two ML models
- Saves trained models + evaluation visuals
- Generates WordClouds to compare language in real vs fake news

---

## Models Used

| Model               | Accuracy | F1 Score |
|--------------------|----------|----------|
| Naive Bayes         | 93.54%   | 93.21%   |
| Logistic Regression | 98.26%   | 98.02%   |

Logistic Regression significantly outperformed Naive Bayes in both accuracy and F1 score.

---

## 📷 Sample Visuals

### Confusion Matrix (Logistic Regression)
![Confusion Matrix](outputs/confusion_matrix_logistic_regression.png)

### WordClouds  
Fake News | Real News  
:-------------------------:|:-------------------------:  
![Fake WordCloud](outputs/wordcloud_fake.png) | ![Real WordCloud](outputs/wordcloud_real.png)

---

## 🧠 ML & NLP Concepts Covered

- Supervised Learning (Binary Classification)
- Text Cleaning & Preprocessing
- TF-IDF Vectorization
- Naive Bayes & Logistic Regression
- Model Evaluation (Precision, Recall, F1 Score)
- Data Visualization (WordClouds, Confusion Matrix)

---

## How to Run the Project

### 🔧 Install dependencies
```bash
pip install -r requirements.txt
```

### Download dataset
Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
Place `Fake.csv` and `True.csv` into the `data/` folder.

### Run full pipeline
```bash
python main.py
```

### Or run notebook interactively
```bash
notebooks/Fake_News_Detection.ipynb
```

---

## Future Work

- Add deep learning models (LSTM, BERT)
- Deploy via Streamlit or Flask
- Extend to multiclass classification (e.g. topic labeling)

---

## License

MIT License. Free to use and modify with credit.

---

Built by Aashish Panthi as part of a Machine Learning final project.
