import os,errno
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, send_file,session
from flask_bootstrap import Bootstrap
from Summarizer import Summarizer
from Lang import Lang
import sys

app = Flask(__name__)
app.debug = True
app.secret_key = os.urandom(24)
bootstrap = Bootstrap(app)


@app.route('/')
def landingPage():
    UserData = {}
    UserData["summary"] = ""
    UserData["input_text"] = ""
    return render_template("index.html",**UserData)

@app.route('/summarize', methods=['POST'])
def index():
    UserData = {}
    su = Summarizer()
    if request.method == "POST":

        input_text = request.form['input_text']

        try:
            summary = su.get_summary(input_text)
            print("Summary: ",summary)
        except:
            summary = sys.exc_info()

        UserData['summary'] = summary
        UserData["input_text"] = input_text

    su = None
    return render_template('index.html', **UserData)


if __name__=='__main__':
    app.run(debug=True,port=3000)
