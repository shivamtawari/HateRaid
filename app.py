import os

import pandas as pd
from flask import Flask, request, render_template, redirect
from predict import Model
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = os.path.join('static', 'img', 'memes')
img_path = None
cap = None
pred = None
last = None

model = Model()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

df = pd.DataFrame(columns=['fname', 'lname', 'email', 'img_path', 'type', 'reason'])


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

def reset():
    global img_path
    global cap
    global last
    if img_path:
        last = img_path
        img_path = None
    else:
        last = cap
        cap = None

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    global img_path
    global cap

    reset()
    if request.method == 'POST':
        f = request.files['image']
        if f == '':
            f = None
            img_path = None

        cap = request.form.get('caption')
        print(cap)
        if cap == '':
            cap = None

        if (f):
            img_path = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'], f.filename)
            f.save(img_path)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        return redirect('/prediction')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    global img_path
    global cap
    global pred
    if (img_path and cap):
        print('In image and caption')
        pred = model.predict(image_dir=img_path, text=cap)
        return render_template('prediction.html', pred=int(pred), img=img_path, text='No')

    elif (img_path):
        print('In image')
        pred = model.predict(image_dir=img_path)
        print(pred)
        return render_template('prediction.html', pred=int(pred), img=img_path, text='No')

    else:
        print('Textttt')
        pred = model.predict(text=cap)
        img_path = None
        return render_template('prediction.html', pred=int(pred), img='No', text=cap)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    global pred
    global img_path
    global cap
    global last
    fname = request.form.get('fname')
    lname = request.form.get('lname')
    email = request.form.get('email')
    reason = request.form.get('reason')
    if img_path==None:
        save = last
    else:
        save = last
    ## save these in database with ~pred and img_path
    df.loc[len(df.index)] = [fname, lname, email, save, int(pred==0), reason]
    df.to_csv('feedback.csv')

    return redirect('/')

if __name__ == '__main__':
    #from werkzeug.serving import run_simple
    #run_simple('localhost', 5000, app)
    app.run(debug=True)