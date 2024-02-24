import nltk
import json
import pickle
import numpy as np
import random
import tensorflow
from data_preprocessing import get_stem_words

ignore_words = ['?', '!', ',', '.', "'s", "'m"]
model = tensorflow.keras.models.load_model("./chatbot_model.h5")
intents = json.loads(
    open("PRO-C120-Teacher-Boilerplate-Code-main/intents.json").read())
words = pickle.load(open("./words.pkl", "rb"))
classes = pickle.load(open("./classes.pkl", "rb"))


def pre_process_user_input(userInput):
    token1 = nltk.word_tokenize(userInput)
    token2 = get_stem_words(token1, ignore_words)
    token2 = sorted(list(set(token2)))
    bag = []
    bow = []
    for word in words:
        if word in token2:
            bow.append(1)
        else:
            bow.append(0)
    bag.append(bow)
    return (np.array(bag))


def botclasspredition(userInput):
    inp = pre_process_user_input(userInput)
    prediction = model.predict(inp)
    predictedClassLabel = np.argmax(prediction[0])
    return (predictedClassLabel)


def botResponse(userInput):
    predictedClassLabel = botclasspredition(userInput)
    predictedClass = classes[predictedClassLabel]
    for intent in intents["intents"]:
        if intent["tag"] == predictedClass:
            botResponse = random.choice(intent["responses"])
            return (botResponse)


print("Hello, I am Stella, how can I help you?")
while True:
    userInput = input("Type your message here: ")
    print("User input:", userInput)
    response = botResponse(userInput)
    print("Bot response: ", response)
