import sys
import os,errno,gc
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, send_file,session
from flask_bootstrap import Bootstrap

CWD = os.getcwd()
sys.path.append('{}/src'.format(CWD))

from Summarizer import Summarizer
from Lang import Lang

app = Flask(__name__)
app.debug = True
app.secret_key = os.urandom(24)
bootstrap = Bootstrap(app)

global su
su = None

@app.route('/')
def landingPage():
    global su
    su = None
    gc.collect()
    su = Summarizer()
    UserData = {}
    UserData["summary"] = ""
    UserData["input_text"] = ""
    return render_template("index.html",**UserData)

@app.route('/summarize', methods=['POST'])
def index():

    gc.collect()
    global su
    UserData = {}
    
    if request.method == "POST":

        input_text = ""
        expected_summary = ""

        # try:
        input_text = request.form['input_text'].strip()
        expected_summary = request.form['expected_summary'].strip()
        
        summary = su.get_summary(input_text)

        if expected_summary is not None and len(expected_summary) > 0:
            flag = su.update_model(input_text,expected_summary)

        # except:
        #     summary = sys.exc_info()

        UserData["input_text"] = input_text
        UserData['expected_summary'] = expected_summary
        UserData['summary'] = summary

        print("Input text {}".format(input_text))
        print("Expected summary text {}".format(expected_summary))
        print("Summary text {}".format(summary))

    return render_template('index.html', **UserData)


if __name__=='__main__':
    app.run(debug=True,port=3000)
