Got it 👍 Thanks for pointing this out. The problem is just formatting — the project structure block was breaking out of the flow. I’ll fix it so the **entire README is inside one clean markdown block** (no broken formatting, no code fences ending too early).

Here’s the fully fixed **copy-paste `README.md`**:

```markdown
# YouTube Sentiment Analysis (Chrome Extension + Flask API)

A full-stack **YouTube Sentiment Analysis system** built as a **Chrome Extension** connected to a **Flask API backend**.  
The project analyzes YouTube video comments, predicts sentiment (Positive/Negative), and visualizes results with charts, trend graphs, and word clouds.

---

## 🚀 Features

- **Chrome Extension** – fetches comments from any YouTube video directly in the browser.
- **Flask API Backend** – processes text with NLP pipeline and returns predictions.
- **Visualizations** – sentiment distribution pie chart, monthly trend graphs, and comment wordcloud.
- **ML Model** – XGBoost + TF-IDF Vectorizer for sentiment classification.
- **Experiment Tracking** – DVC + MLflow for dataset, model, and experiment management.
- **Automation** – CI/CD workflow with GitHub Actions.

---

## 🧩 Project Components

### 1. Chrome Extension (Frontend)

Located in `chrome_plugin/`

- Fetches comments via **YouTube Data API**
- Sends comments to Flask backend for predictions
- Displays:
  - Metrics (total comments, unique commenters, average comment length, average sentiment score)
  - Sentiment Pie Chart
  - Sentiment Trend Graph
  - Wordcloud
  - Top 25 comments with sentiment labels

### 2. Flask API (Backend)

Located in `flask_api/main.py`

- REST Endpoints:
  - `/predict` → Sentiment prediction
  - `/predict_with_timestamps` → Prediction + timestamp
  - `/generate_chart` → Pie chart of sentiment
  - `/generate_wordcloud` → Wordcloud from comments
  - `/generate_trend_graph` → Sentiment trend over time
- Loads pretrained **XGBoost sentiment model** + **TF-IDF vectorizer**

---

## 📂 Project Structure
```

.
├── .github/workflows/cicd.yaml # CI/CD pipeline
├── chrome_plugin/ # Chrome extension files
│ ├── manifest.json
│ ├── popup.html
│ └── popup.js
├── data/ # Data (DVC managed)
├── flask_api/ # Flask backend
│ └── main.py
├── notebooks/ # Jupyter notebooks for experiments
├── senti/ # NLP helper code
├── src/ # Source code for training and utils
├── confusion_matrix.png
├── Dockerfile
├── dvc.yaml / dvc.lock / .dvcignore # DVC files
├── model_registration.py
├── params.yaml
├── requirements.txt
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
└── xgb_model.pkl # Trained XGBoost model

````

---

## ⚡ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis
````

### 2. Setup backend (Flask API)

```bash
cd flask_api
pip install -r ../requirements.txt
python main.py
```

Runs on: `http://localhost:5000`

### 3. Setup Chrome Extension

1. Open Chrome → Extensions → Manage Extensions → Enable Developer Mode
2. Click **Load unpacked** and select the `chrome_plugin/` folder
3. Open any YouTube video → click the extension icon → see sentiment insights

---

## 📸 Screenshots

_(add your extension popup and charts here)_

---

## 📜 License

MIT License

```

---

✅ Now the **project structure stays inside the same markdown block**.
✅ The whole file is **one clean copy-paste chunk**.

Do you also want me to add a **“How It Works (Step by Step)”** section (fetch → preprocess → predict → visualize), or keep it minimal like this?
```
