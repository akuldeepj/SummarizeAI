import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate, TimeDistributed
from tensorflow.keras.models import Model

# Load the CNN/DailyMail dataset
import os
import random

# Set the path to the folder containing the stories
data_dir = "D:\SummarizeAI\cnn_stories\cnn\stories"

# Set the percentage of stories to use for each set
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

# Get a list of all the story files in the folder
all_stories = [f for f in os.listdir(data_dir) if f.endswith(".story")]

# Shuffle the list of story files
random.shuffle(all_stories)

# Split the stories into train, validation, and test sets
train_stories = all_stories[:int(len(all_stories) * train_percent)]
val_stories = all_stories[int(len(all_stories) * train_percent):int(len(all_stories) * (train_percent + val_percent))]
test_stories = all_stories[int(len(all_stories) * (train_percent + val_percent)):]

# Print the number of stories in each set
print("Number of train stories:", len(train_stories))
print("Number of validation stories:", len(val_stories))
print("Number of test stories:", len(test_stories))


# Define parameters
MAX_TEXT_LEN = 500
MAX_SUMMARY_LEN = 50
VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

# Load the text data
def load_text_data(stories):
    texts = []
    for file in stories:
        with open(os.path.join(data_dir, file), "r",encoding="utf-8") as f:
            text = f.read().replace("\n", " ")
            texts.append(text)
    return texts

def load_summary_data(stories):
    summaries = []
    for file in stories:
        with open(os.path.join(data_dir, file[:-6] + ".summary"), "r") as f:
            summary = f.read().replace("\n", " ")
            summaries.append(summary)
    return summaries


train_texts = load_text_data(train_stories)
val_texts = load_text_data(val_stories)
test_texts = load_text_data(test_stories)

# Load the summary data


train_summaries = load_summary_data(train_stories)
val_summaries = load_summary_data(val_stories)
test_summaries = load_summary_data(test_stories)

# Tokenize the text and summary data
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_seq = tokenizer.texts_to_sequences(train_texts)
val_seq = tokenizer.texts_to_sequences(val_texts)
test_seq = tokenizer.texts_to_sequences(test_texts)

tokenizer.fit_on_texts(train_summaries)

train_summary_seq = tokenizer.texts_to_sequences(train_summaries)
val_summary_seq = tokenizer.texts_to_sequences(val_summaries)
test_summary_seq = tokenizer.texts_to_sequences(test_summaries)

# Pad the text and summary sequences
train_padded_seq = pad_sequences(train_seq, maxlen=MAX_TEXT_LEN, padding="post", truncating="post")
val_padded_seq = pad_sequences(val_seq, maxlen=MAX_TEXT_LEN, padding="post", truncating="post")
test_padded_seq = pad_sequences(test_seq, maxlen=MAX_TEXT_LEN, padding="post", truncating="post")

train_summary_padded_seq = pad_sequences(train_summary_seq, maxlen=MAX_SUMMARY_LEN, padding="post", truncating="post")
val_summary_padded_seq = pad_sequences(val_summary_seq, maxlen=MAX_SUMMARY_LEN, padding="post", truncating="post")
test_summary_padded_seq = pad_sequences(test_summary_seq, maxlen=MAX_SUMMARY_LEN, padding="post", truncating="post")

# Define the model architecture
input_text = Input(shape=(MAX_TEXT_LEN,))
embedding_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_TEXT_LEN)(input_text)
lstm_layer = LSTM(units=HIDDEN_DIM, return_sequences=True)(embedding_layer)
attention_layer = Attention()([lstm_layer, lstm_layer])
dense_layer_1 = Dense(units=1, activation="tanh")(attention_layer)
dense_layer_2 = Dense(units=MAX_SUMMARY_LEN, activation="softmax")(dense_layer_1)
output_summary = Concatenate(axis=1)([dense_layer_2, dense_layer_2])
model = Model(inputs=input_text, outputs=output_summary)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_padded_seq, to_categorical(train_summary_padded_seq, num_classes=VOCAB_SIZE), 
                    validation_data=(val_padded_seq, to_categorical(val_summary_padded_seq, num_classes=VOCAB_SIZE)), 
                    epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_padded_seq, to_categorical(test_summary_padded_seq, num_classes=VOCAB_SIZE))
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
