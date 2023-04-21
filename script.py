import flask
import os
from flask import send_from_directory, request, render_template
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
PS = PorterStemmer()
from string import punctuation
punctuation = list(punctuation)
punctuation.append("'s")
punctuation.append("'m")
punctuation.append("'d")
punctuation.append("'ve")
import json
import random
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

app = flask.Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/favicon.png')

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

def preprocess(text):
    tokenized_words = nltk.word_tokenize(text)
    tokenized_words = [word.lower() for word in tokenized_words]
    cleaned_words = [word for word in tokenized_words if word not in punctuation]
    stemmed_words = [PS.stem(word) for word in cleaned_words]
    return " ".join(stemmed_words)

labelled_data = {0: 'greeting', 1: 'name', 2: 'goodbye', 3: 'address', 4: 'course', 5: 'information', 6: 'library', 7: 'fees', 8: 'hours', 9: 'canteen', 10: 'scholarship', 11: 'projects', 12: 'admission', 13: 'hostel', 14: 'events', 15: 'placements', 16: 'clubs'}


@app.route('/webhook', methods=['POST','GET'])
def webhook():
    req = request.get_json(force=True)
    prompt = req['queryResult']['queryText']
    p = loaded_model.decision_function([preprocess(prompt)])
    print(p)
    ma = p[0][np.argmax(p)]
    mi = p[0][np.argmin(p)]
    print(np.argmax(p))
    print(labelled_data[np.argmax(p)])
    with open('intents.json') as file:
        data = json.load(file)
    d = data['intents'][np.argmax(p)]
    responses = d['responses']
    answer = random.sample(responses,k=1)[0]
    print(ma,mi)
    if ma <= -1 and mi > -2:
        answer = "Sorry Didn't Get You!"
    return {
        'fulfillmentText': f'{answer}'
    }

if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run()