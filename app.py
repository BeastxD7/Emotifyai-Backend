from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire app (allow requests from localhost:3000)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load tokenizer and model (similar to your current setup)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_dict = {0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 
                5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment', 
                10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 
                14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 
                19: 'nervousness', 20: 'optimism', 21: 'pride', 22: 'realization', 
                23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral', 
                28: 'other_emotions'}

# Define the BERTClass
class BERTClass(torch.nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super(BERTClass, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, n_classes)

    def forward(self, ids, mask):
        hidden_state = self.bert(input_ids=ids, attention_mask=mask)[0]
        pooled_output = hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def predict(self, sentence):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        max_len = 40
        inputs = tokenizer(sentence, add_special_tokens=True, max_length=max_len,
                           padding="max_length", truncation=True, return_tensors="pt")

        ids = inputs['input_ids'].to(DEVICE)
        mask = inputs['attention_mask'].to(DEVICE)

        self.eval()
        logits = self(ids=ids, mask=mask)
        result = torch.sigmoid(logits)  # Probabilities for all classes
        return result

# Load the saved model
model_path = 'saved_model'
model = torch.load(model_path, map_location=DEVICE)
model.to(DEVICE)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Emotion Detection API. Use the /predict endpoint to get predictions."

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    if 'sentence' not in data:
        return jsonify({'error': 'Missing "sentence" in request body'}), 400

    sentence = data['sentence']

    # Get predictions
    predictions = model.predict(sentence)
    probabilities = predictions[0].tolist()  # Convert tensor to list

    # Apply threshold and filter emotions
    threshold = 0.3
    emotions_above_threshold = {emotion_dict[idx]: prob for idx, prob in enumerate(probabilities) if prob > threshold}

    return jsonify({
        'input_sentence': sentence,
        'probabilities': {emotion_dict[idx]: prob for idx, prob in enumerate(probabilities)},
        'filtered_emotions': emotions_above_threshold
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)