from flask import Flask,render_template,request
import pickle
from textblob import TextBlob

app = Flask(__name__)
filename = 'svm_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))

def split_into_stems(message):
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.stem() for word in words]

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        message =split_into_stems(message)
        message = ' '.join(message)
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
    
