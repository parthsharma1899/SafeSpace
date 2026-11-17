from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# Define model path relative to the project directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "stress_classifier_epoch2")

# Load model and tokenizer once
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # Set to eval mode

def predict_stress_score(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        stress_score = probs[0][1].item()  # Score for "Stressed"
    return round(stress_score, 4)

