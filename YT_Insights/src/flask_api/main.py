import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import nltk
from pathlib import Path

nltk.download("wordnet")

app = Flask(__name__)
CORS(app)

def preprocess_comment(comment: str):

    comment = comment.lower()
    comment = comment.strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", '', comment)

    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment


def load_model_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("")
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    project_root = Path(__file__).resolve().parents[2]
    vectorizer_file = project_root / vectorizer_path

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as error:
        print("failed to load model from mlflow ")
    #     local_model_file = project_root / "trained_model" / "lgbm_model.pkl"
    #     if not local_model_file.exists():
    #         raise RuntimeError(
    #             f"Could not load registered model {model_uri} and local fallback is missing"
    #         ) from error
    #     with local_model_file.open("rb") as file:
    #         model = pickle.load(file)

    with vectorizer_file.open('rb') as file:
        vectorizer = pickle.load(file)

    return model, vectorizer

model, vect = load_model_vectorizer(
    model_name="yt_chrome_plugin_model",
    model_version="2",
    vectorizer_path="trained_model/tfidf_vctorizer.pkl"
)


@app.route("/")
def home():
    return "Welcome to the YT sentiment plugin"

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "no comments provided"}), 400
    try:

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vect.transform(preprocessed_comments)
        model_input = pd.DataFrame(
            transformed_comments.toarray(),
            columns=vect.get_feature_names_out()
        )
        # dense_comments = transformed_comments.toarray()

        predictions = model.predict(model_input).tolist()

    except Exception as e:
        return jsonify({"error": f"prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


@app.route("/predict_with_timestamps", methods = ['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error":"No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vect.transform(preprocessed_comments)
        model_input = pd.DataFrame(
                    transformed_comments.toarray(),
                    columns=vect.get_feature_names_out()
                )
        
        
        predictions = model.predict(model_input).tolist()

        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500
        
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)



@app.route("/generate_chart", methods = ['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']

        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment count sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            colors=colors,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis("equal")
        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"char generation failed {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
