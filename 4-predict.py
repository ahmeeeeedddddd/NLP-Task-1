import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model and tokenizer
model = tf.keras.models.load_model("model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
    p = model.predict(pad, verbose=0)[0][0]
    return "positive" if p > 0.5 else "negative", float(p)

print(predict("this movie was great and the acting was wonderful"))
print(predict("terrible film not worth watching"))
