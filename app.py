from flask import Flask, request, jsonify
from flask_cors import CORS
from detoxify import Detoxify
from textblob import TextBlob
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


detox_model = Detoxify('original')

app = Flask(__name__)


CORS(app)


def detect_and_mask_toxicity(text):
    words = text.split()
    masked_words = []
    for word in words:
        toxicity_scores = detox_model.predict(word)  
        
        if any(score > 0.5 for score in toxicity_scores.values()):
            masked_words.append('[REDACTED]')
        else:
            masked_words.append(word)
    return ' '.join(masked_words)


def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        user_input = request.json.get('text', '')
        if not user_input:
            return jsonify({"error": "No input text provided."}), 400

        
        cleaned_text = detect_and_mask_toxicity(user_input)

        
        sentiment = analyze_sentiment(cleaned_text)

        return jsonify({
            "cleaned_text": cleaned_text,
            "sentiment": sentiment
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
