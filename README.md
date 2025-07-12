# 📰 Spam News Detection using NLP and Multinomial Naive Bayes

This project focuses on detecting fake or spam news articles using Natural Language Processing (NLP) techniques and the Multinomial Naive Bayes classifier. It involves data preprocessing, feature extraction with TF-IDF, and classification of news articles as real or fake.

---

## 📌 Project Objectives

- Classify news articles as real or fake using NLP.
- Preprocess raw text data to extract meaningful patterns.
- Apply TF-IDF to convert text into numerical features.
- Train and evaluate a Multinomial Naive Bayes model.
- Visualize results and interpret model performance.

---

## 🧠 Methodology

1. **Data Collection**  
   - Two datasets: `True.csv` and `Fake.csv`, each containing news articles labeled accordingly.

2. **Preprocessing**  
   - Removing punctuation, stopwords, and non-alphabetic characters  
   - Tokenization and lemmatization using NLTK

3. **Feature Extraction**  
   - TF-IDF vectorization to convert text data into numerical features

4. **Model Training**  
   - Trained using the Multinomial Naive Bayes classifier from scikit-learn

5. **Evaluation**  
   - Accuracy Score  
   - Confusion Matrix  
   - Word Count Visualizations

---

## 📊 Results & Visualizations

- Achieved high classification accuracy with balanced class performance.
- Confusion matrix used to evaluate true vs. false predictions.
- Word count plot highlighted the most frequent words in fake and real news articles.
- TF-IDF combined with Multinomial Naive Bayes showed reliable performance with lightweight training.

---

## 📁 File Structure
📦 SpamNewsDetection
├── Fake.csv
├── True.csv
├── SpamNew.ipynb
└── README.md


---

## 🧪 Libraries Used

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `matplotlib`
- `seaborn`

---

## 🚀 How to Run

1. Clone the repository:
git clone https://github.com/rahul1842/Spam_News.git
cd Spam_News


2. Install required packages:
pip install -r requirements.txt


3. Open and run `SpamNew.ipynb` in Jupyter Notebook.

---

## 🧾 License

This project is open source and available under the [MIT License](LICENSE).

---

THANK YOU!!!!!!!!!
