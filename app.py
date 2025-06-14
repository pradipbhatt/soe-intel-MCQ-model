
from flask import Flask, render_template, request, send_file, session, url_for, redirect
from flask_bootstrap import Bootstrap
import spacy
import random
from collections import Counter
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# TensorFlow/Keras imports for LSTM-based MCQ generation
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

app = Flask(__name__)
app.secret_key = 'your_secret_key'
Bootstrap(app)

# Load spaCy model with word vectors (using medium model for vectors)
nlp = spacy.load("en_core_web_md")

# ---------------------------
# LSTM-based MCQ Generation Functions
# ---------------------------
def preprocess_text(text):
    """Split the text into sentences using spaCy."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def create_training_data(sentences, tokenizer, max_length):
    """Convert sentences into padded numerical sequences."""
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

def build_lstm_model(vocab_size, max_length, embedding_dim):
    """Build and compile an LSTM model for learning sentence structures."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def find_similar_words(word, num_similar=3):
    """Find similar words using spaCy word vectors.
       Returns placeholder distractors if no vector is available."""
    word_token = nlp.vocab[word] if word in nlp.vocab else None
    if not word_token or not word_token.has_vector:
        return ["[Distractor]"] * num_similar

    similarities = []
    for token in nlp.vocab:
        if token.is_alpha and token.has_vector and token != word_token:
            similarity = word_token.similarity(token)
            similarities.append((token.text, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [w for w, sim in similarities[:num_similar]]

def generate_mcqs(text, tokenizer, max_length, model, num_questions=5):
    """
    Generate MCQs using LSTM-based approach combined with spaCy.
    For each randomly selected sentence from the input text, a noun is replaced by a blank.
    Distractor options are generated based on word vector similarities.
    """
    sentences = preprocess_text(text)
    if not sentences:
        return []
    
    selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))
    mcqs = []
    for sentence in selected_sentences:
        doc = nlp(sentence)
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        if len(nouns) < 1:
            continue

        subject = random.choice(nouns)
        question_stem = sentence.replace(subject, "______")

        # Generate distractor words using spaCy word vectors
        similar_words = find_similar_words(subject, num_similar=3)
        answer_choices = [subject] + similar_words
        random.shuffle(answer_choices)
        correct_answer = chr(65 + answer_choices.index(subject))  # 'A', 'B', 'C', or 'D'

        mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs

# ---------------------------
# Global Initialization (Tokenizer and LSTM Model)
# ---------------------------
# A sample text is used for fitting the tokenizer and creating a vocabulary.
sample_text = """Deep learning is a subset of machine learning that uses neural networks. LSTMs are useful for processing sequential data like text. 
Natural language processing involves techniques like tokenization and named entity recognition."""
sentences = preprocess_text(sample_text)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
max_length = 20
model = build_lstm_model(vocab_size, max_length, embedding_dim=100)

# ---------------------------
# Functions for File and URL Processing and PDF Generation
# ---------------------------
def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        text += page_text
    return text

def process_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove unwanted elements
        for script in soup(['script', 'style', 'header', 'footer', 'nav']):
            script.decompose()
        return soup.get_text(separator='\n')
    except Exception as e:
        print(f"Error processing URL: {e}")
        return ""

def draw_multiline_text(pdf, text, x, y, max_width):
    """Draw text on the PDF canvas, wrapping it if it exceeds max_width."""
    lines = []
    words = text.split(" ")
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if pdf.stringWidth(test_line, "Helvetica", 12) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= 14  # Move down for the next line
    return y

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""
        # Check if URL is provided
        if 'url' in request.form and request.form['url']:
            url = request.form['url']
            text = process_url(url)
        # Check if manual text is provided
        elif 'manual_text' in request.form and request.form['manual_text']:
            text = request.form['manual_text']
        # Check if files were uploaded
        elif 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += file.read().decode('utf-8')
                    
        num_questions = int(request.form['num_questions'])
        mcqs = generate_mcqs(text, tokenizer, max_length, model, num_questions=num_questions)
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
        session['mcqs'] = mcqs_with_index
        # Redirect to mcqs page after generation
        return render_template('mcqs.html', mcqs=mcqs_with_index)
    return render_template('index.html')

@app.route('/result')
def result():
    mcqs = session.get('mcqs', [])
    return render_template('result.html', mcqs=mcqs)

@app.route('/download_pdf')
def download_pdf():
    mcqs = session.get('mcqs', [])
    if not mcqs:
        return "No MCQs to download.", 400  # Handle no MCQs case

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf.setFont("Helvetica", 12)

    y_position = height - 40
    margin = 30
    max_width = width - 2 * margin

    for index, mcq in mcqs:
        question, choices, correct_answer = mcq
        y_position = draw_multiline_text(pdf, f"Q{index}: {question}?", margin, y_position, max_width)
        options = ['A', 'B', 'C', 'D']
        for i, choice in enumerate(choices):
            y_position = draw_multiline_text(pdf, f"{options[i]}: {choice}", margin + 20, y_position, max_width)
        pdf.drawString(margin + 20, y_position, f"Correct Answer: {correct_answer}")
        y_position -= 20
        if y_position < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y_position = height - 40

    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='mcqs.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)

 