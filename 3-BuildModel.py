import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

#enabling gpu memory growth to avoid any errors in memory allocation
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import json


df = pd.read_csv("imdb_preprocessed.csv")
df["final_text"] = df["final_text"].fillna("")

df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

empty = (df["final_text"].str.strip() == "")
print("Empty reviews:", empty.sum())
df = df[~empty].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["final_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

VOCAB_SIZE = 20000

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_seq = tokenizer.texts_to_sequences(X_train)
test_seq = tokenizer.texts_to_sequences(X_test)

print(X_train.iloc[0][:80])
print(train_seq[0][:15])

lengths = [len(s) for s in train_seq]
print("Average:", np.mean(lengths), "| 95th percentile:", np.percentile(lengths, 95))

MAX_LEN = 200

X_train_pad = pad_sequences(train_seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
X_test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
print(X_train_pad.shape)

model = Sequential([
    Embedding(VOCAB_SIZE, 64,mask_zero=True),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
              metrics=["accuracy"])
model.summary()

history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1)

loss, acc = model.evaluate(X_test_pad, y_test)
print(f"Test accuracy: {acc:.2%}")

def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
    p = model.predict(pad)[0][0]
    return "positive" if p > 0.5 else "negative", float(p)

print(predict("this movie was great and the acting was wonderful"))
print(predict("terrible film not worth watching"))

model.save("model.keras")
model.save_weights("model_weights.weights.h5")

accuracies = {
    "train_accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"],
    "test_accuracy": float(acc)
}
with open("accuracies.json", "w") as f:
    json.dump(accuracies, f, indent=4)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
