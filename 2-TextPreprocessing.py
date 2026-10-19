import re
import string
import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

data = pd.read_csv("imdb_cleaned.csv")

data["clean"] = data["review"].str.replace(r"<.*?>", " ", regex=True)

data["clean"] = data["clean"].str.lower()

data["clean"] = data["clean"].str.translate(str.maketrans("", "", string.punctuation))#removes: !"#$%&'()*+,-./:;<=>?@[\]^_{|}~.

data["clean"] = data["clean"].apply(lambda x: re.sub(r"\s+", " ", x).strip())

print(data[["review", "clean"]].head(3))

data["tokens"] = data["clean"].apply(word_tokenize)

stop_words = set(stopwords.words("english"))
stop_words -= {"no", "not", "nor"}   # choosing what not to remove

data["tokens"] = data["tokens"].apply(lambda toks: [w for w in toks if w not in stop_words])

lemmatizer = WordNetLemmatizer()
data["tokens"] = data["tokens"].apply(lambda toks: [lemmatizer.lemmatize(w) for w in toks])

print(data[["review", "tokens"]].head(3))

data["final_text"] = data["tokens"].apply(" ".join)
data[["final_text", "sentiment"]].to_csv("imdb_preprocessed.csv", index=False)