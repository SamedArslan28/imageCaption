from IPython.display import display
from PIL import Image
import pickle
from keras.applications import VGG16
from random import shuffle, seed
from keras import models
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils import pad_sequences
import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense, Embedding, LSTM, add
import os
import pickle
from keras.models import load_model

image_dic = {}
with open('./data/captions.txt') as f:
    lines = f.readlines()
    for i in lines:
        sentences = i.split(",")
        if sentences[0] in image_dic.keys():
            image_dic[sentences[0]].append(i[len(sentences[0]) + 1:-3])
        else:
            image_dic[sentences[0]] = [i[len(sentences[0]) + 1:-3]]


def show_image_and_captions(img_id):
    data = "data/images/" + img_id
    display(Image.open("./data/images/" + img_id))
    print("--------------------------Captions are--------------------------")
    for description in range(len(image_dic[img_id])):
        print(image_dic[img_id][description])


image_names = list(image_dic.keys())
seed(42)
shuffle(image_names)

split_1 = int(0.8 * len(image_names))
train_filenames = image_names[:split_1]
test_filenames = image_names[split_1:]


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption


clean(image_dic)

modelvgg = VGG16(include_top=True)
modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-2].output)

target_size = [224, 224, 3]

all_captions = []
for key in image_dic:
    for caption in image_dic[key]:
        all_captions.append(caption)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
tokenizer.fit_on_texts(all_captions)
max_length = max(len(caption.split()) for caption in all_captions)

inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

with open('features.pkl', 'rb') as f:
    images = pickle.load(f)
def idx_to_word(intgr, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == intgr:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq '
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break

    return in_text


model = load_model('traninedModel3.h5')


def show_image_and_pred(image_name):
    y_prd = predict_caption(model, images[image_name], tokenizer, max_length)
    return y_prd


import streamlit as st
from PIL import Image

col1, col2, col3 = st.columns([1, 2, 1])

col1.header("Welcome to Image Caption API")

photo_uploader = col2.file_uploader("Upload a Photo")

if photo_uploader:
    st.success("Uploaded Successfully")
    prediction = show_image_and_pred(photo_uploader.name)
    col2.markdown("Predicted sentence is:" + prediction[:-6].capitalize())

    st.image(photo_uploader)
