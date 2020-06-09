import tensorflow as tf
from flask import Flask, render_template, request
from keras.models import load_model, Sequential




app = Flask(__name__)


@app.route('/')
def index(): 
    return '''
    <html>
    <body>
<a href=/fileUpload>file upload</a>
    </body>
    </html>
    '''

@app.route('/upload', methods = ['POST'])
def upload():
    






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)