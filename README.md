# 📊 CyberSentiment

**Social Media Sentiment Analysis for Early Detection of Cybersecurity Threat Trends**

## 🎯 Project Aim

The goal of this project is to develop a system that collects cybersecurity-related discussions from social media platforms (Twitter and Reddit), applies sentiment analysis and machine learning techniques, and provides early warning indicators of potential cybersecurity threats.

By leveraging public discourse, the system aims to support **Security Operations Centers (SOCs)** and analysts with **real-time situational awareness**, reducing the risk of undetected emerging threats.

---

## Project Structure

cybersentiment/
│── data/                  # Raw and processed datasets
│   ├── raw/               # Original scraped CSV/JSON
│   ├── processed/         # Cleaned/preprocessed data
│
│── scripts/               # Data collection + preprocessing scripts
│   ├── twitter_scraper.py
│   ├── reddit_scraper.py
│   ├── preprocess.py
│
│── models/                # ML & DL models (saved .pkl/.h5)
│   ├── vader_baseline.pkl (optional for ML/DL only)
│   ├── logistic_model.pkl
│   ├── cnn_model.h5
│
│── notebooks/             # Jupyter notebooks for experiments
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│
│── cyber-sentiment/
├── app.py
├── config.py
├── models.py
├── forms.py
├── requirements.txt
├── instance/
│ └── config.py # secret keys (ignored by git)
├── static/
│ ├── css/
│ │ └── styles.css
│ ├── js/
│ │ └── main.js
│ └── vendor/
│ └── chart.min.js # Chart.js (or use CDN)
├── templates/
│ ├── base.html
│ ├── index.html
│ ├── register.html
│ ├── login.html
│ ├── dashboard.html
│ └── alerts.html
├── utils/
│ └── sentiment_utils.py # model loading + helper functions
├── data/
│ └── sample_posts.csv # optional sample data
└── README.md
│
│── tests/                 # Unit tests for scrapers, models, API
│
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── .gitignore             # Ignore venv, data, cache, etc.


## 🧰 Tech Stack

**Data Collection**

* Python (Tweepy, PRAW, Requests, Pandas)

**Preprocessing & NLP**

* NLTK, spaCy, Regex, Scikit-learn

**Sentiment & Threat Modeling**

* VADER (lexicon-based baseline)
* Scikit-learn (Logistic Regression, XGBoost with TF-IDF)
* Keras/TensorFlow (CNN, BiLSTM)

**Backend & Model Serving**

* Django (Python web framework)
* Django REST Framework (API layer)
* SQLite / PostgreSQL (database)

**Frontend**

* HTML5, CSS3, Bootstrap 5
* JavaScript (Chart.js / D3.js for visualizations)

**Deployment**

* Docker (containerization)
* Heroku / AWS / Render (cloud hosting)

---

## 🛠 Planned Development Roadmap

### **Phase 1: Data Collection (✅ in progress)**

* Collect data from **Twitter (API v2)** and **Reddit (Pushshift + PRAW)**.
* Store posts with metadata (platform, text, timestamp, sentiment placeholder).
* Run scripts manually (automation to be added later).

### **Phase 2: Data Preprocessing**

* Clean and normalize text (remove URLs, emojis, special characters).
* Tokenization, stopword removal, lemmatization.
* Store processed data separately from raw data.

### **Phase 3: Sentiment & Threat Modeling**

* Implement **VADER** baseline for quick sentiment scoring.
* Train **classical ML model** (Logistic Regression / XGBoost) with TF-IDF.
* Train **lightweight deep learning model** (CNN / BiLSTM).
* Evaluate models on accuracy, precision, recall, and F1-score.

### **Phase 4: Model Integration**

* Build a prediction pipeline (VADER → ML → DL).
* Store results (sentiment & threat classification) in database.

### **Phase 5: Web Application Development**

* Develop a **Django-based web app**.
* Frontend with **HTML, CSS, Bootstrap 5, JavaScript**.
* Features:

  * Dashboard with sentiment trends.
  * Keyword search & filtering.
  * Alerts for high-risk posts.

### **Phase 6: Deployment & Testing**

* Deploy web app on cloud (Heroku / AWS / Render).
* Implement continuous monitoring & periodic retraining.

---

📌 This roadmap will be **updated as development progresses**, with details added for each phase once implemented.

---