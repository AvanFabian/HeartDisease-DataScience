from flask import Flask, request, jsonify

app = Flask(__name__)
# helloworld
@app.route('/')
def hello_world():
    return 'Hello, World!'
