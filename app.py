from flask import Flask, request, jsonify, render_template_string
import pickle
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("MNB.pkl", "rb"))
cv    = pickle.load(open("cv.pkl", "rb"))
ps    = PorterStemmer()

def preprocess(message):
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower().split()
    review = [ps.stem(w) for w in review if w not in stopwords.words('english')]
    return ' '.join(review)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Spam Detector</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Arial, sans-serif;
            background: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .card {
            background: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 500px;
            text-align: center;
        }
        h2 { font-size: 24px; margin-bottom: 8px; color: #1a1a2e; }
        p  { color: #666; margin-bottom: 24px; font-size: 14px; }
        textarea {
            width: 100%;
            padding: 14px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 15px;
            resize: vertical;
            outline: none;
            transition: border 0.2s;
        }
        textarea:focus { border-color: #6c63ff; }
        button {
            margin-top: 16px;
            width: 100%;
            padding: 14px;
            background: #6c63ff;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #574fd6; }
        .result {
            margin-top: 20px;
            padding: 16px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            display: none;
        }
        .spam    { background: #ffe0e0; color: #c0392b; }
        .not-spam { background: #e0ffe0; color: #27ae60; }
    </style>
</head>
<body>
    <div class="card">
        <h2>📧 Email Spam Detector</h2>
        <p>Paste any email or message below to check if it is spam</p>
        <textarea id="msg" rows="5" placeholder="Type or paste your message here..."></textarea>
        <button onclick="predict()">Check Message</button>
        <div id="result" class="result"></div>
    </div>
    <script>
        async function predict() {
            const msg = document.getElementById('msg').value.trim();
            if (!msg) { alert('Please enter a message!'); return; }
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg })
            });
            const data = await res.json();
            const box = document.getElementById('result');
            box.style.display = 'block';
            if (data.result === 'SPAM') {
                box.className = 'result spam';
                box.innerText = '🚨 This message is SPAM!';
            } else {
                box.className = 'result not-spam';
                box.innerText = '✅ This message is NOT Spam';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json()
    message = data['message']
    cleaned = preprocess(message)
    vector  = cv.transform([cleaned]).toarray()
    result  = model.predict(vector)[0]
    return jsonify({"result": "SPAM" if result else "NOT SPAM"})

if __name__ == '__main__':
    app.run(debug=True)
