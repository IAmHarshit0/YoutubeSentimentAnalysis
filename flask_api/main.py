# main.py
import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from wordcloud import WordCloud
import numpy as np
import pickle
import traceback
import re

app = Flask(__name__)
CORS(app)

# -------------------- Helpers -------------------- #
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text

def normalize_prediction(pred):
    """Force prediction to 0 or 1"""
    if isinstance(pred, (int, np.integer)):
        return int(pred)
    s = str(pred).lower().strip()
    if s in ["1", "positive"]:
        return 1
    return 0

# -------------------- Model Loading -------------------- #
try:
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Model and vectorizer loaded.")
except Exception as e:
    print("❌ Failed to load model/vectorizer:", e)
    model, vectorizer = None, None

SENTIMENT_LABELS = {0: "Negative", 1: "Positive"}

# -------------------- Routes -------------------- #
@app.route("/")
def home():
    return "✅ Sentiment API running on /predict and /predict_with_timestamps"

@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    try:
        data = request.get_json(force=True)
        comments = data.get("comments", [])
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        texts = [clean_text(c.get("text", "")) for c in comments]
        timestamps = [c.get("timestamp", "") for c in comments]

        if model is None or vectorizer is None:
            return jsonify({"error": "Model or vectorizer not loaded"}), 500

        X = vectorizer.transform(texts)
        preds = model.predict(X).tolist()
        preds = [normalize_prediction(p) for p in preds]

        response = []
        for orig, ts, pred in zip(texts, timestamps, preds):
            response.append({
                "comment": orig,
                "timestamp": ts,
                "sentiment": pred,
                "sentiment_label": SENTIMENT_LABELS[pred]
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Prediction failed", "trace": traceback.format_exc()}), 500

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json(force=True)
        counts = data.get("sentiment_counts", {})

        pos = int(counts.get("1", 0)) + int(counts.get(1, 0)) + int(counts.get("Positive", 0))
        neg = int(counts.get("0", 0)) + int(counts.get(0, 0)) + int(counts.get("Negative", 0))

        if pos + neg == 0:
            return jsonify({"error": "No sentiment counts"}), 400

        plt.figure(figsize=(4, 4))  # Reduced from (5, 5)
        plt.pie([pos, neg], labels=["Positive", "Negative"],
                colors=["#4CAF50", "#FF5252"],
                autopct="%1.1f%%", 
                startangle=140,
                textprops={'color': 'white', 'fontweight': 'bold'},
                pctdistance=0.85)
        plt.axis("equal")

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", transparent=True)
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype="image/png")
    except Exception:
        return jsonify({"error": "Chart generation failed", "trace": traceback.format_exc()}), 500

@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json(force=True)
        comments = data.get("comments", [])
        text = " ".join([clean_text(c) for c in comments])

        if not text.strip():
            return jsonify({"error": "No text for wordcloud"}), 400

        wc = WordCloud(width=800, height=400, background_color="black", colormap="coolwarm").generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)

        return send_file(buf, mimetype="image/png")
    except Exception:
        return jsonify({"error": "Wordcloud generation failed", "trace": traceback.format_exc()}), 500

@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json(force=True)
        sentiment_data = data.get("comments", [])

        if not sentiment_data:
            return jsonify({"error": "No comments provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sentiment'] = df['sentiment'].astype(int)

        # Set the timestamp as the index and sort
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Resample by day and calculate sentiment percentages
        daily_counts = df.resample('D')['sentiment'].value_counts().unstack(fill_value=0)
        daily_totals = daily_counts.sum(axis=1)
        daily_percentages = (daily_counts.T / daily_totals).T * 100

        # Ensure all sentiment columns exist
        for sentiment in [0, 1]:
            if sentiment not in daily_percentages.columns:
                daily_percentages[sentiment] = 0

        # Create the plot
        plt.figure(figsize=(8, 4))
        colors = {
            0: "#FF5252",  # Negative - matching your existing color scheme
            1: "#4CAF50"   # Positive - matching your existing color scheme
        }

        for sentiment in [0, 1]:
            plt.plot(
                daily_percentages.index,
                daily_percentages[sentiment],
                marker='o',
                markersize=4,
                linestyle='-',
                linewidth=2,
                label=SENTIMENT_LABELS[sentiment],
                color=colors[sentiment]
            )

        plt.title('Sentiment Trends Over Time', color='white', pad=15)
        plt.xlabel('Date', color='white', labelpad=10)
        plt.ylabel('Percentage of Comments (%)', color='white', labelpad=10)
        
        # Style the plot
        plt.grid(True, alpha=0.2)
        plt.gca().set_facecolor('#1a1b1e')  # Match your UI background
        plt.gcf().set_facecolor('#1a1b1e')
        
        # Style the axes
        plt.gca().spines['bottom'].set_color('#2d2d2d')
        plt.gca().spines['top'].set_color('#2d2d2d')
        plt.gca().spines['right'].set_color('#2d2d2d')
        plt.gca().spines['left'].set_color('#2d2d2d')
        
        # Style the ticks
        plt.tick_params(colors='white')
        plt.xticks(rotation=45)
        
        # Format dates on x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))

        # Style the legend
        plt.legend(facecolor='#1a1b1e', edgecolor='#2d2d2d', labelcolor='white')
        
        plt.tight_layout()

        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", facecolor='#1a1b1e', edgecolor='none')
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype="image/png")
    except Exception:
        return jsonify({"error": "Trend graph generation failed", "trace": traceback.format_exc()}), 500

# -------------------- Run -------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
