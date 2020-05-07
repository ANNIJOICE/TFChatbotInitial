import flask
from flask import request
from flask_socketio import SocketIO, emit
import nltk

# Do this in your ipython notebook or analysis script
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

import pickle
import tflearn
import nltk

# Do this in your ipython notebook or analysis script
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import random

# import our chat-bot intents file

app = flask.Flask(__name__)
app.config["DEBUG"] = True
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/train', methods=['GET'])
def train_data():
    if 'name' in request.args:
        botName = str(request.args['name'])
        import json 
        with open(botName+'.json') as json_data:
            intents = json.load(json_data)
        with open('entity.json') as entity_data:
            entity = json.load(entity_data)
        words = []
        classes = []
        documents = []
        ignore_words = ['?']
        # loop through each sentence in our intents patterns
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                w = word_tokenize(pattern)
                entity_list = []
                # add to our words list
                words.extend(w)
                for entityWord in w:
                    for en in entity["entity"]:
                        if en["tag"] == entityWord:
                            entity_list = en["value"]

                w.extend(entity_list)
                # add to documents in our corpus
                documents.append((w, intent['tag']))
                # add to our classes list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
        for en in entity['entity']:
            for value in en["value"]: 
                words.append(value)
        # stem and lower each word and remove duplicates
        words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))

        # remove duplicates
        classes = sorted(list(set(classes)))

        # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(classes)

        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            entityList = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])




        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            entityList = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            for i,p_words in enumerate(doc[0]):
                for en in entity['entity']:
                    if p_words == en["tag"]:
                        for word in en["value"]:
                            pattern_words[i] = word
                            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
                            bag = []
                            for w in words:
                                bag.append(1) if w in pattern_words else bag.append(0)
                            output_row = list(output_empty)
                            output_row[classes.index(doc[1])] = 1
                            print([bag, output_row], "bag")
                            training.append([bag, output_row])
                            print(training)


        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists
        train_x = list(training[:,0])
        train_y = list(training[:,1])

        # reset underlying graph data
        tf.reset_default_graph()
        # Build neural network
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        model = tflearn.DNN(net, tensorboard_dir= botName+'_logs')
        # Start training (apply gradient descent algorithm)
        model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
        model.save(botName+'.tflearn')
        # save all of our data structures
        import pickle
        pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( botName+"_training_data", "wb" ) )
        print(tflearn.variables.get_all_trainable_variable ())
        return "Trained Successfully"
    
    else:
        return "Error: No Bot name provided"

@app.route('/response', methods=['GET'])
def response_data():
    if 'name' in request.args:  
        botName = str(request.args['name'])
        sentence = str(request.args['userInput'])
        data = pickle.load( open( botName+"_training_data", "rb" ) )
        words = data['words']
        classes = data['classes']
        train_x = data['train_x']
        train_y = data['train_y']
        import json 
        with open(botName+'.json') as json_data:
            intents = json.load(json_data)
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        model = tflearn.DNN(net, tensorboard_dir=botName+'_logs')
        # load our saved model
        model.load('./'+botName+'.tflearn')
        def clean_up_sentence(sentence):
            # tokenize the pattern
            sentence_words = nltk.word_tokenize(sentence)
            # stem each word
            sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
            return sentence_words

        # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
        def bow(sentence, words, show_details=False):
            # tokenize the pattern
            sentence_words = clean_up_sentence(sentence)
            # bag of words
            bag = [0]*len(words)  
            for s in sentence_words:
                for i,w in enumerate(words):
                    if w == s: 
                        bag[i] = 1
                        if show_details:
                            print ("found in bag: %s" % w)
            print(bag,"bag")
            return(np.array(bag))

        # create a data structure to hold user context
        context = {}

        ERROR_THRESHOLD = 0.25
        def classify(sentence):
            # generate probabilities from the model
            print(words)
            results = model.predict([bow(sentence, words)])[0]
            # print(results)
            # filter out predictions below a threshold
            results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
            # sort by strength of probability
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append((classes[r[0]], r[1]))
            # return tuple of intent and probability
            return return_list

        def response(sentence, userID='123', show_details=False):
            results = classify(sentence)
            print(results)
            # if we have a classification then find the matching intent tag
            if results:
                # loop as long as there are matches to process
                while results:
                    for i in intents['intents']:
                        # find a tag matching the first result
                        if i['tag'] == results[0][0]:
                            # set context for this intent if necessary
                            if 'context_set' in i:
                                if show_details: print ('context:', i['context_set'])
                                context[userID] = i['context_set']

                            # check if this intent is contextual and applies to this user's conversation
                            if not 'context_filter' in i or \
                                (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                                if show_details: print ('tag:', i['tag'])
                                # a random response from the intent
                                print(random.choice(i['responses']))
                                return random.choice(i['responses'])

                    results.pop(0)
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print ('context:', i['context_set'])
                            context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                            if show_details: print ('tag:', i['tag'])
                                # a random response from the intent
                            print(random.choice(i['responses']))
                            return random.choice(i['responses']) 
        return response(sentence) 
    else:
        return "Error: No Bot name provided"

@socketio.on('my chat', namespace='/socket')
def socket_response_data(botData):
    print(botData['userInput'])
    if 'name' in botData:  
        botName = str(botData['name'])
        sentence = str(botData['userInput'])
        data = pickle.load( open( botName+"_training_data", "rb" ) )
        words = data['words']
        classes = data['classes']
        train_x = data['train_x']
        train_y = data['train_y']
        import json 
        with open(botName+'.json') as json_data:
            intents = json.load(json_data)
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        model = tflearn.DNN(net, tensorboard_dir=botName+'_logs')
        # load our saved model
        model.load('./'+botName+'.tflearn')
        def clean_up_sentence(sentence):
            # tokenize the pattern
            sentence_words = nltk.word_tokenize(sentence)
            # stem each word
            sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
            return sentence_words

        # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
        def bow(sentence, words, show_details=False):
            # tokenize the pattern
            sentence_words = clean_up_sentence(sentence)
            # bag of words
            bag = [0]*len(words)  
            for s in sentence_words:
                for i,w in enumerate(words):
                    if w == s: 
                        bag[i] = 1
                        if show_details:
                            print ("found in bag: %s" % w)
            print(bag,"bag")
            return(np.array(bag))

        # create a data structure to hold user context
        context = {}

        ERROR_THRESHOLD = 0.25
        def classify(sentence):
            # generate probabilities from the model
            print(words)
            results = model.predict([bow(sentence, words)])[0]
            # print(results)
            # filter out predictions below a threshold
            results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
            # sort by strength of probability
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append((classes[r[0]], r[1]))
            # return tuple of intent and probability
            return return_list

        def response(sentence, userID='123', show_details=False):
            results = classify(sentence)
            print(results)
            # if we have a classification then find the matching intent tag
            if results:
                # loop as long as there are matches to process
                while results:
                    for i in intents['intents']:
                        # find a tag matching the first result
                        if i['tag'] == results[0][0]:
                            # set context for this intent if necessary
                            if 'context_set' in i:
                                if show_details: print ('context:', i['context_set'])
                                context[userID] = i['context_set']

                            # check if this intent is contextual and applies to this user's conversation
                            if not 'context_filter' in i or \
                                (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                                if show_details: print ('tag:', i['tag'])
                                # a random response from the intent
                                print(random.choice(i['responses']))
                                return random.choice(i['responses'])

                    results.pop(0)
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print ('context:', i['context_set'])
                            context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                            if show_details: print ('tag:', i['tag'])
                                # a random response from the intent
                            print(random.choice(i['responses']))
                            return random.choice(i['responses']) 
        # retu response(sentence) 
        emit('my response', response(sentence))
    else:
        emit('my response', 'done')


socketio.run(app)