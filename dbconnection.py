#!flask/bin/python
from flask import Flask, jsonify
from pymongo import MongoClient

app = Flask(__name__)
client = MongoClient("mongodb://127.0.0.1:27017")  # host uri
db = client.mymongodb  # Select the database
data_collection = db.flowers  # Select the collection name
initial_data = [data for data in data_collection.find()]

if (len(initial_data)) == 0:
    data_collection.insert({
        "tag": "greeting",
        "patterns": ["hi", "How are you", "Is anyone there?", "hello", "Good day"],
        "responses": ["Hello, Good to see you again", "Hi there, how can I help?"],
        "context_set": ""
          })
    data_collection.insert({
        "tag": "flowers",
        "patterns": ["need flowers","is flower available", "flower"],
        "responses": ["yes we have, How much bouque do u want"],
        "context_set": ""
          })
    data_collection.insert({
        "tag": "flower_set",
        "patterns": ["need number bouque"],
        "responses": ["Ok mam, Here is ur bouque"],
        "context_set": ""
          })
    data_collection.insert({
        "tag": "goodbye",
        "patterns": ["Thank you", "Bye"],
        "responses": ["Visit again mam! Thank u"],
        "context_set": ""
          })

if __name__ == '__main__':
    app.run(debug=True)