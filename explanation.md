# Complete Technical & Conceptual Study Guide: NLP Sentiment Analysis

This document is an exhaustive, line-by-line educational study guide for the IMDb Sentiment Analysis NLP pipeline. It covers every file, every line of code, the syntax, the underlying data science and deep learning theory, and the architectural decisions made throughout the project.

---

## Table of Contents
1. [End-to-End Project Architecture Overview](#end-to-end-project-architecture-overview)
2. [Section 1: Data Cleaning (`1-Dataclean.py`)](#section-1-data-cleaning-1-datacleanpy)
   - [Line-by-Line Code Breakdown & Syntax](#1-line-by-line-breakdown)
   - [Core Concepts & Purpose](#1-core-concepts--purpose)
3. [Section 2: Text Preprocessing & NLP Pipeline (`2-TextPreprocessing.py`)](#section-2-text-preprocessing--nlp-pipeline-2-textpreprocessingpy)
   - [Line-by-Line Code Breakdown & Syntax](#2-line-by-line-breakdown)
   - [Core NLP Concepts & Purpose](#2-core-nlp-concepts--purpose)
4. [Section 3: Model Architecture, Training & Regularization (`3-BuildModel.py`)](#section-3-model-architecture-training--regularization-3-buildmodelpy)
   - [Line-by-Line Code Breakdown & Syntax](#3-line-by-line-breakdown)
   - [Deep Dive: Natural Language Representation Concepts](#3-deep-dive-representation-concepts)
   - [Deep Dive: Neural Network Architecture & The LSTM](#3-deep-dive-lstm-architecture)
   - [Deep Dive: Training Dynamics, Overfitting & Early Stopping](#3-deep-dive-training-dynamics)
   - [Deep Dive: Metrics (Train vs. Validation vs. Test)](#3-deep-dive-metrics)
5. [Section 4: Production Inference & Prediction Pipeline (`4-predict.py`)](#section-4-production-inference--prediction-pipeline-4-predictpy)
   - [Line-by-Line Code Breakdown & Syntax](#4-line-by-line-breakdown)
   - [Inference Workflow & Production Considerations](#4-inference-workflow)
6. [Quick Reference Glossary of Key Terms](#quick-reference-glossary)

---

# End-to-End Project Architecture Overview

The objective of this project is **Binary Sentiment Classification**: determining whether an English movie review expresses a **positive** or **negative** opinion.

The project is structured into four sequential, decoupled stages:

```
[Raw Dataset: imdb_messy.csv]
           │
           ▼
┌──────────────────────────────────────┐
│       1-Dataclean.py                 │  --> Removes null records & duplicate rows
└──────────────────────────────────────┘
           │
           ▼ (imdb_cleaned.csv)
┌──────────────────────────────────────┐
│       2-TextPreprocessing.py         │  --> HTML strip, lowercase, punctuation removal,
└──────────────────────────────────────┘      stopwords filtering, lemmatization
           │
           ▼ (imdb_preprocessed.csv)
┌──────────────────────────────────────┐
│       3-BuildModel.py                │  --> Tokenization, sequence padding, Embedding,
└──────────────────────────────────────┘      LSTM layer, Dropout, EarlyStopping, Model Saving
           │
           ▼ (model.keras, model_weights.weights.h5, tokenizer.pkl, accuracies.json)
┌──────────────────────────────────────┐
│       4-predict.py                   │  --> Loads saved artifacts and predicts sentiment
└──────────────────────────────────────┘      on new raw text reviews
```

---

# Section 1: Data Cleaning (`1-Dataclean.py`)

The first stage cleans tabular anomalies (missing values, duplicated records) before applying any natural language algorithms.

## 1. Line-by-Line Breakdown

### Lines 1–3: Imports
```python
1: from PIL import GimpGradientFile
2: import pandas as pd
3: import numpy as np
```
- **Line 1 (`from PIL import GimpGradientFile`)**:
  - *What it does*: Attempts to import an image gradient utility from the Pillow library.
  - *Syntax*: Standard Python `from <module> import <object>` statement.
  - *Why it is here*: This is an accidental auto-import (often generated automatically by IDE auto-complete while typing). It is unused in the script and has no operational effect on data cleaning.
- **Line 2 (`import pandas as pd`)**:
  - *What it does*: Imports the Pandas library and assigns it the standard alias `pd`.
  - *Why we did it*: Pandas provides DataFrame structures, vectorised file I/O (`read_csv`, `to_csv`), and tabular transformation tools essential for data manipulation.
- **Line 3 (`import numpy as np`)**:
  - *What it does*: Imports NumPy, aliased as `np`.
  - *Why we did it*: Pandas is built on top of NumPy arrays; NumPy provides underlying numerical representations and `NaN` (Not a Number) handling.

---

### Line 5: Loading the Raw Data
```python
5: data = pd.read_csv("imdb_messy.csv")
```
- *Syntax*: `pd.read_csv(filepath)` parses a comma-separated values file into a 2D tabular `DataFrame`.
- *What we get out of it*: The variable `data` holds the table of raw reviews and sentiment labels in memory.
- *Why we did it*: The raw data must be loaded into memory before inspections and transformations can occur.

---

### Lines 7–9: Exploratory Data Inspection
```python
7: print(data.shape)
8: print(data.head())
9: data.info()
```
- **Line 7 (`data.shape`)**:
  - *Syntax*: Attribute access returning a tuple: `(number_of_rows, number_of_columns)`.
  - *Purpose*: Tells us the volume of data immediately upon loading.
- **Line 8 (`data.head()`)**:
  - *Syntax*: Method returning the first 5 rows of the DataFrame.
  - *Purpose*: Allows manual inspection of column names, data format, and text style (e.g. observing HTML tags, quotation marks, or emojis).
- **Line 9 (`data.info()`)**:
  - *Syntax*: Method printing index range, column datatypes, non-null counts, and RAM footprint.
  - *Purpose*: Quickly reveals if columns contain missing values and verifies datatypes (e.g., text columns as `object`).

---

### Line 11: Visual Divider
```python
11: print(60*"/")
```
- *Syntax*: String repetition operator (`"string" * integer`).
- *Purpose*: Prints `////////////////////////////////////////////////////////////` in the console to visually separate diagnostic output sections during execution.

---

### Lines 13–15: Detecting Null (Missing) Values
```python
13: print(data.isnull().sum())
15: print("Total rows with a null value: ", data.isnull().any(axis=1).sum())#axis=1: will go row by row
```
- **Line 13 (`data.isnull().sum()`)**:
  - *Syntax*: `data.isnull()` yields a boolean DataFrame of identical dimensions (`True` where a cell is `NaN`/`None`). `.sum()` aggregates down columns (`axis=0` by default), counting total nulls per column.
  - *Purpose*: Identifies which columns have missing data (e.g., missing review text vs. missing labels).
- **Line 15 (`data.isnull().any(axis=1).sum()`)**:
  - *Syntax*: `any(axis=1)` checks each row horizontally across all columns. If at least one column in that row has a null value, it returns `True`. `.sum()` counts how many rows have at least one missing field.
  - *Purpose*: Tells us exactly how many complete rows will be dropped if listwise deletion is applied.

---

### Lines 17–20: Dropping Missing Records
```python
17: before = len(data)
18: print("Before dropping null values: ",before)
19: data = data.dropna()
20: print(f"Dropped {before - len(data)} rows with a null value, Remaining: {len(data)}")
```
- **Line 17 (`before = len(data)`)**:
  - *Syntax*: Stores the integer row count before modification.
- **Line 19 (`data = data.dropna()`)**:
  - *Syntax*: Returns a new DataFrame omitting every row that contains one or more `NaN` values.
  - *Why we did it*: Machine learning algorithms cannot handle `NaN` inputs. A review without text provides no signal, and a review without a label cannot be used for supervised training.
- **Line 20**:
  - *Syntax*: Formatted f-string calculating the delta `before - len(data)` and printing remaining rows.
  - *Purpose*: Provides explicit confirmation and auditing of how many rows were purged.

---

### Lines 22–26: Detecting and Removing Duplicates
```python
22: before = len(data)
23: print("Duplicated rows:", data.duplicated().sum())
24: data = data.drop_duplicates()
25: data = data.reset_index(drop=True)
26: print(f"Dropped {before - len(data)} duplicates, Remaining: {len(data)}")
```
- **Line 23 (`data.duplicated().sum()`)**:
  - *Syntax*: `data.duplicated()` returns a boolean Series marking rows that are identical duplicates of an earlier row. `.sum()` totals them.
- **Line 24 (`data = data.drop_duplicates()`)**:
  - *Syntax*: Removes identical repeated rows, keeping the first occurrence.
  - *Why we did it*: If duplicate reviews are present, they can leak into both the training set and testing set (data leakage). This causes artificially inflated evaluation scores and biases the model toward repeated samples.
- **Line 25 (`data = data.reset_index(drop=True)`)**:
  - *Syntax*: Re-indexes the DataFrame from `0` to `N-1`. `drop=True` discards the old fragmented index rather than saving it as a new column.
  - *Why we did it*: Dropping rows leaves holes in the index (e.g., indices `0, 1, 4, 7...`). Resetting ensures contiguous numeric addressing.

---

### Line 28: Exporting Cleaned Data
```python
28: data.to_csv("imdb_cleaned.csv", index=False)
```
- *Syntax*: Serialises the DataFrame to a CSV file. `index=False` prevents Pandas from writing the row index numbers as an unwanted leading column (`Unnamed: 0`).
- *Why we did it*: Saves the cleaned dataset to disk so subsequent NLP steps can load it without repeating the null/duplicate cleaning steps.

---

# Section 2: Text Preprocessing & NLP Pipeline (`2-TextPreprocessing.py`)

Raw human text is noisy: it contains HTML markup, casing differences, punctuation, useless functional grammar words (stopwords), and inflected word forms. This script normalizes text into clean linguistic tokens.

## 2. Line-by-Line Breakdown

### Lines 1–8: Imports
```python
1: import re
2: import string
3: import nltk
4: import pandas as pd
6: from nltk.tokenize import word_tokenize
7: from nltk.corpus import stopwords
8: from nltk.stem import WordNetLemmatizer
```
- **Line 1 (`import re`)**:
  - *Purpose*: Python's built-in Regular Expressions module. Used for pattern-based string substitutions (removing HTML tags and collapsing whitespace).
- **Line 2 (`import string`)**:
  - *Purpose*: Contains pre-defined string constants, specifically `string.punctuation` (which includes `!"#$%&'()*+,-./:;<=>?@[\]^_{|}~`).
- **Line 3 (`import nltk`)**:
  - *Purpose*: Natural Language Toolkit, the standard Python NLP library providing linguistic rules, corpora, and lexicons.
- **Line 6 (`from nltk.tokenize import word_tokenize`)**:
  - *Purpose*: Imports NLTK's standard sentence-to-word tokenizer, which breaks sentences into individual word tokens based on punctuation and grammar boundaries.
- **Line 7 (`from nltk.corpus import stopwords`)**:
  - *Purpose*: Imports curated dictionaries of common grammatical filler words (e.g., "is", "the", "at", "which").
- **Line 8 (`from nltk.stem import WordNetLemmatizer`)**:
  - *Purpose*: Imports an algorithmic lemmatizer linked to Princeton's WordNet lexical database to reduce inflected words to their root lemma.

---

### Line 10: Loading the Cleaned Data
```python
10: data = pd.read_csv("imdb_cleaned.csv")
```
- *Purpose*: Loads the dataset produced by `1-Dataclean.py`.

---

### Line 12: Removing HTML Markup
```python
12: data["clean"] = data["review"].str.replace(r"<.*?>", " ", regex=True)
```
- *Syntax*:
  - `.str.replace(pat, repl, regex=True)`: Vectorized string replacement across all rows.
  - `r"<.*?>"`:
    - `<`: Match an opening angle bracket.
    - `.*?`: Non-greedy (lazy) match of any character sequence.
    - `>`: Match a closing angle bracket.
- *Why we did it*: IMDb reviews scraped from the web contain HTML formatting tags like `<br /><br />` or `<i>`. These tags carry no sentiment information and would otherwise be treated as real words by our vocabulary. Replacing them with a space `" "` ensures adjacent words do not accidentally merge together.

---

### Line 14: Lowercasing
```python
14: data["clean"] = data["clean"].str.lower()
```
- *Syntax*: Converts all alphabetical characters to lowercase.
- *Why we did it*: Computers treat `"Great"`, `"great"`, and `"GREAT"` as three completely different tokens with distinct memory addresses. Lowercasing normalizes them to a single token `"great"`, shrinking vocabulary size and pooling semantic statistics.

---

### Line 16: Punctuation Removal
```python
16: data["clean"] = data["clean"].str.translate(str.maketrans("", "", string.punctuation))
```
- *Syntax*:
  - `str.maketrans("", "", string.punctuation)`: Creates a translation mapping table where every character in `string.punctuation` maps to `None` (deletion).
  - `.str.translate(...)`: Applies this mapping across every string in the column.
- *Why we did it*: Strips symbols such as commas, periods, quotes, and exclamation points so that `"movie."` becomes `"movie"`. This prevents punctuation from sticking to words and splitting their identity.

---

### Line 18: Normalizing Whitespace
```python
18: data["clean"] = data["clean"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
```
- *Syntax*:
  - `re.sub(r"\s+", " ", x)`: Replaces any consecutive sequence of whitespace characters (spaces, tabs, newlines `\n`, carriage returns `\r`) with a single space `" "`.
  - `.strip()`: Removes any leading and trailing spaces from the string.
- *Why we did it*: HTML removal and punctuation stripping leave behind erratic spaces and dangling newlines. This standardizes the review into clean, single-space-separated sentences.

---

### Line 22: Word Tokenization
```python
22: data["tokens"] = data["clean"].apply(word_tokenize)
```
- *Syntax*: Maps `word_tokenize` over every string, transforming `"good movie"` into Python list of strings: `["good", "movie"]`.
- *Why we did it*: Natural Language algorithms cannot process monolithic text strings directly. Text must be cut into atomic linguistic units (tokens) before stopword filtering and lemmatization can operate on them.

---

### Lines 24–27: Selective Stopword Removal
```python
24: stop_words = set(stopwords.words("english"))
25: stop_words -= {"no", "not", "nor"}   # choosing what not to remove
27: data["tokens"] = data["tokens"].apply(lambda toks: [w for w in toks if w not in stop_words])
```
- *Syntax*:
  - `stopwords.words("english")`: Loads the default list of English stopwords from NLTK.
  - `set(...)`: Converts the list into a Python `set` for $O(1)$ constant-time lookup performance.
  - `stop_words -= {"no", "not", "nor"}`: Set subtraction operator. Removes negation words from the stopword set.
  - `[w for w in toks if w not in stop_words]`: List comprehension filtering out any token found in the `stop_words` set.
- *Why we did it*:
  - Words like "the", "a", "is", "in" appear in high volumes across all documents but provide zero discriminatory information regarding whether someone liked or disliked a film. Removing them reduces sequence lengths and eliminates noise.
  - **Crucial Design Decision (`stop_words -= {"no", "not", "nor"}`)**: In sentiment analysis, negation words are the most critical sentiment-flipping words in the language. If you strip `"not"`, the sentence `"this is not good"` becomes `"good"`, completely reversing the label from negative to positive! Keeping negations is vital for sentiment accuracy.

---

### Lines 29–30: Lemmatization
```python
29: lemmatizer = WordNetLemmatizer()
30: data["tokens"] = data["tokens"].apply(lambda toks: [lemmatizer.lemmatize(w) for w in toks])
```
- *Syntax*: Instantiates `WordNetLemmatizer` and calls `.lemmatize(w)` on each token in the list.
- *What it does*: Reduces inflected words to their base dictionary form (lemma). For example: `"movies"` $\to$ `"movie"`, `"acting"` $\to$ `"act"`, `"characters"` $\to$ `"character"`.
- *Why Lemmatization over Stemming?*
  - **Stemming** (e.g. Porter Stemmer) is a crude heuristic that chops off word suffixes, often resulting in non-words (e.g., `"crying"` $\to$ `"cri"`, `"universe"` $\to$ `"univers"`).
  - **Lemmatization** uses morphological analysis and vocabulary databases to ensure words are transformed into valid root words.

---

### Lines 34–35: Reconstructing and Saving Preprocessed Data
```python
34: data["final_text"] = data["tokens"].apply(" ".join)
35: data[["final_text", "sentiment"]].to_csv("imdb_preprocessed.csv", index=False)
```
- **Line 34 (`" ".join`)**:
  - *Syntax*: Joins the clean token list back into a single whitespace-delimited string (e.g., `["good", "script"]` $\to$ `"good script"`).
  - *Why we did it*: Deep learning tokenizers (like Keras's `Tokenizer`) expect text strings as input, from which they internally manage sequence indexing and vocabulary tables.
- **Line 35 (`to_csv`)**:
  - *Syntax*: Subsets the DataFrame to keep only the feature column `final_text` and target column `sentiment`, exporting them to `imdb_preprocessed.csv`.

---

# Section 3: Model Architecture, Training & Regularization (`3-BuildModel.py`)

This is the core machine learning file. It converts preprocessed text into numerical tensors, feeds them into an LSTM Deep Learning architecture, trains the model with regularization and early stopping, and exports the model artifacts.

## 3. Line-by-Line Breakdown

### Lines 1–2: Silencing TensorFlow Verbose Logging
```python
1: import os
2: os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
```
- *Syntax*: Modifies the operating system environment variables before importing TensorFlow.
- *Values*:
  - `"0"` = All logs (INFO, WARNING, ERROR).
  - `"1"` = Filters out INFO logs.
  - `"2"` = Filters out INFO and WARNING logs (shows only ERROR).
  - `"3"` = Filters out all logs including errors.
- *Why we did it*: Prevents C++ backend hardware messages (like AVX instruction notices and memory alerts) from flooding the terminal console.

---

### Lines 6–13: GPU Memory Growth Configuration
```python
7: gpus = tf.config.list_physical_devices("GPU")
8: if gpus:
9:     try:
10:         for gpu in gpus:
11:             tf.config.experimental.set_memory_growth(gpu, True)
12:     except RuntimeError as e:
13:         print(e)
```
- *Syntax*: Queries physical hardware devices detected by TensorFlow. If a CUDA-compatible GPU is available, it iterates through each device and enables `memory_growth`.
- *Why we did it*: By default, TensorFlow attempts to allocate nearly 100% of the GPU's VRAM upon initialization. If multiple processes run or if the allocation exceeds hardware boundaries, CUDA throws an out-of-memory error. Enabling memory growth ensures TensorFlow allocates only the memory needed at each step, growing dynamically.

---

### Lines 15–25: Core Imports
```python
15: import pandas as pd
16: import numpy as np
17: from sklearn.model_selection import train_test_split
18: from tensorflow.keras.preprocessing.text import Tokenizer
19: from tensorflow.keras.preprocessing.sequence import pad_sequences
20: from tensorflow.keras.models import Sequential
21: from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
22: from tensorflow.keras.optimizers import Adam
23: from tensorflow.keras.callbacks import EarlyStopping
24: import pickle
25: import json
```
- `train_test_split`: Scikit-learn utility for splitting arrays/DataFrames into randomized training and test partitions with stratification.
- `Tokenizer`: Vectorizes text corpora into integer sequences where each integer represents a unique word.
- `pad_sequences`: Transforms variable-length integer sequences into uniform 2D matrices by adding padding tokens.
- `Sequential`: Keras model container for stacking linear layers where data flows sequentially from input to output.
- `Embedding`: Dense vector lookup table layer mapping sparse word indices into continuous coordinate space.
- `LSTM`: Long Short-Term Memory recurrent neural layer designed for sequential data.
- `Dense`: Fully connected neural layer where each neuron connects to all activations from the preceding layer.
- `Dropout`: Regularization layer that randomly zeroes activations during training to prevent co-adaptation.
- `Adam`: Adaptive Moment Estimation optimization algorithm for updating network weights via gradient descent.
- `EarlyStopping`: Training callback that halts training when validation loss stops improving.
- `pickle` & `json`: Standard libraries for serializing objects to disk.

---

### Lines 28–35: Data Ingestion and Cleansing of Preprocessed Text
```python
28: df = pd.read_csv("imdb_preprocessed.csv")
29: df["final_text"] = df["final_text"].fillna("")
31: df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
33: empty = (df["final_text"].str.strip() == "")
34: print("Empty reviews:", empty.sum())
35: df = df[~empty].reset_index(drop=True)
```
- **Line 29 (`.fillna("")`)**: Fills any blank cells with an empty string so string operations do not raise exceptions.
- **Line 31 (`.map(...)`)**: Converts categorical sentiment strings (`"positive"`, `"negative"`) into binary integers: **1 for positive, 0 for negative**. Neural networks cannot compute error losses on raw strings; labels must be numeric.
- **Lines 33–35**: Some reviews might consist solely of stopwords or punctuation (e.g. `"<br />...?"`). After preprocessing in Script 2, these become empty strings `""`. These rows are filtered out using boolean masking (`~empty`) to ensure every row contains meaningful text.

---

### Lines 37–39: Stratified Train/Test Split
```python
37: X_train, X_test, y_train, y_test = train_test_split(
38:     df["final_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
39: )
```
- *Syntax*:
  - `test_size=0.2`: 80% of data is allocated to training (~30,789 samples), 20% to testing (~7,698 samples).
  - `random_state=42`: Seeds the pseudo-random number generator, ensuring identical data partitions across runs for reproducibility.
  - `stratify=df["label"]`: **Crucial parameter**. Guarantees that the proportion of positive and negative reviews in both splits exactly matches the original dataset distribution (50% positive / 50% negative), avoiding class imbalance.

---

### Lines 41–44: Tokenizer Initialization and Fitting
```python
41: VOCAB_SIZE = 20000
43: tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
44: tokenizer.fit_on_texts(X_train)
```
- *Syntax*:
  - `VOCAB_SIZE = 20000`: Sets the ceiling on how many unique words the model will remember. The tokenizer counts all word frequencies and assigns indices $1$ to $20000$ to the top 20,000 most frequent words.
  - `oov_token="<OOV>"`: Specifies the Out-Of-Vocabulary placeholder. Any word encountered that is not in the top 20,000 is mapped to this special token rather than being completely ignored.
  - `tokenizer.fit_on_texts(X_train)`: **Must only be called on `X_train`!** Computes word frequency statistics across the training set to build the vocabulary dictionary (`word_index`).
  - *Data Leakage Warning*: Never fit the tokenizer on the test data! In real-world deployment, the model will see unseen words from new users. Fitting only on `X_train` simulates this realistic scenario.

---

### Lines 46–47: Text-to-Sequence Conversion
```python
46: train_seq = tokenizer.texts_to_sequences(X_train)
47: test_seq = tokenizer.texts_to_sequences(X_test)
```
- *What it does*: Replaces every word in a review with its integer index from the vocabulary.
- *Example*:
  - Sentence: `"great movie script"`
  - Sequence: `[12, 17, 542]`

---

### Lines 52–58: Sequence Length Analysis and Padding
```python
52: lengths = [len(s) for s in train_seq]
53: print("Average:", np.mean(lengths), "| 95th percentile:", np.percentile(lengths, 95))
55: MAX_LEN = 200
57: X_train_pad = pad_sequences(train_seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
58: X_test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
```
- **Lines 52–53**: Computes review length distribution. The 95th percentile is ~313 words, and the average is ~122 words.
- **Line 55 (`MAX_LEN = 200`)**: Choosing 200 covers the vast majority of review information while preventing excessive computational load.
- **Lines 57–58 (`pad_sequences`)**:
  - Neural network input tensors must have uniform rectangular dimensions: `(batch_size, sequence_length)`.
  - If a review has fewer than 200 tokens: padded with zeros (`0`).
  - If a review has more than 200 tokens: truncated to 200.
  - `padding="pre"`: Prepends zeros to the *beginning* of the sequence (`[0, 0, ..., 12, 17]`).
  - `truncating="pre"`: Cuts off words from the *beginning*, keeping the final 200 words.
  - *Why "pre" padding for LSTMs?* LSTMs read tokens sequentially from left to right. If padding is placed at the end (`post`), the LSTM processes the real words first, and then processes tens or hundreds of useless zero tokens right before making its prediction. This can dilute the hidden state. With `pre` padding, the LSTM processes zeros first, and the most recent information in its memory when outputting predictions is the actual review text!

---

### Lines 61–66: Neural Network Architecture Definition
```python
61: model = Sequential([
62:     Embedding(VOCAB_SIZE, 64, mask_zero=True),
63:     LSTM(64),
64:     Dropout(0.5),
65:     Dense(1, activation="sigmoid"),
66: ])
```
This is the heart of the neural network. Let's break down each layer:

1. **`Embedding(VOCAB_SIZE, 64, mask_zero=True)`**:
   - `VOCAB_SIZE (20000)`: Size of vocabulary (number of possible word indices).
   - `64`: Dimension of the continuous embedding space. Each word is mapped to a 64-dimensional vector of trainable real numbers.
   - `mask_zero=True`: Tells downstream layers to ignore padded zeros (`0`) so they do not participate in gradient updates or LSTM memory calculations.
   - Output shape: `(batch_size, 200, 64)`.

2. **`LSTM(64)`**:
   - A Long Short-Term Memory recurrent layer with 64 memory cells.
   - It iterates through the 200 time steps sequentially, updating its cell state and hidden state at each step.
   - Output shape: `(batch_size, 64)` — it outputs the final hidden state summarizing the entire sequence.

3. **`Dropout(0.5)`**:
   - Randomly deactivates 50% of the LSTM output activations during each training step.
   - Forces the final Dense neuron not to rely on any single LSTM output feature, preventing co-adaptation and overfitting.

4. **`Dense(1, activation="sigmoid")`**:
   - A single output neuron producing a scalar output between `0.0` and `1.0`.
   - $\text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$.
   - Output interpreted as probability: values close to `1.0` indicate strong positive sentiment; values close to `0.0` indicate negative sentiment.

---

### Lines 68–71: Model Compilation
```python
68: model.compile(loss="binary_crossentropy",
69:               optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
70:               metrics=["accuracy"])
71: model.summary()
```
- **`loss="binary_crossentropy"`**:
  - The standard objective function for two-class classification.
  - Formula: $\mathcal{L} = - [y \log(p) + (1 - y) \log(1 - p)]$.
  - Heavily penalizes confident wrong predictions (e.g., predicting 0.99 for a negative review produces a massive loss).
- **`optimizer=Adam(learning_rate=0.001, clipnorm=1.0)`**:
  - Adam automatically adjusts learning rates for each parameter based on first and second moments of the gradients.
  - `clipnorm=1.0`: **Gradient Clipping**. If the gradient vector norm exceeds 1.0, it is scaled down to 1.0. This prevents the infamous **exploding gradient** problem common in recurrent networks like LSTMs.
- **`metrics=["accuracy"]`**:
  - Monitors the percentage of correct predictions: $\frac{\text{Correct Predictions}}{\text{Total Samples}}$.
- **`model.summary()`**:
  - Prints the architectural table detailing layer names, output dimensions, and trainable parameter counts (totaling 1,313,089 parameters in this model).

---

### Lines 73–74: Training with Early Stopping
```python
73: early_stop = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
74: history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1, callbacks=[early_stop])
```
- **`EarlyStopping(...)`**:
  - `monitor="val_loss"`: Monitors validation loss rather than validation accuracy, because loss is continuous, smooth, and reveals overfitting earlier.
  - `patience=1`: If validation loss fails to reach a new minimum for 1 epoch, training is immediately stopped.
  - `restore_best_weights=True`: Crucial flag. When training halts, the model's weights are automatically restored to the epoch that achieved the lowest validation loss (rather than keeping the overfitted weights of the final epoch).
- **`model.fit(...)`**:
  - `epochs=5`: Maximum number of full passes over the training data.
  - `batch_size=64`: The network processes 64 reviews in parallel before computing gradient descent updates.
  - `validation_split=0.1`: Holds out 10% of the training data (~3,464 reviews) strictly for validation monitoring at the end of each epoch.

---

### Lines 76–77: Final Test Evaluation
```python
76: loss, acc = model.evaluate(X_test_pad, y_test)
77: print(f"Test accuracy: {acc:.2%}")
```
- *Syntax*: Computes loss and accuracy over the unseen `X_test_pad` partition.
- *Purpose*: Provides the final unbiased report of real-world generalization performance (~87.37%).

---

### Lines 79–86: Local Prediction Function & Demonstration
```python
79: def predict(text):
80:     seq = tokenizer.texts_to_sequences([text])
81:     pad = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
82:     p = model.predict(pad)[0][0]
83:     return "positive" if p > 0.5 else "negative", float(p)
85: print(predict("this movie was great and the acting was wonderful"))
86: print(predict("terrible film not worth watching"))
```
- Demonstrates how the trained pipeline processes raw text strings into sentiment predictions.
- Returns a tuple: label (`"positive"` or `"negative"`) and confidence score ($p \in [0, 1]$).

---

### Lines 88–100: Artifact Persistence
```python
88: model.save("model.keras")
89: model.save_weights("model_weights.weights.h5")
91: accuracies = {
92:     "train_accuracy": history.history["accuracy"],
93:     "val_accuracy": history.history["val_accuracy"],
94:     "test_accuracy": float(acc)
95: }
96: with open("accuracies.json", "w") as f:
97:     json.dump(accuracies, f, indent=4)
99: with open("tokenizer.pkl", "wb") as f:
100:    pickle.dump(tokenizer, f)
```
- **Line 88 (`model.save("model.keras")`)**:
  - Saves the complete model architecture, layer configurations, learned weights, and optimizer state in the modern Keras zip archive format.
- **Line 89 (`model.save_weights(...)`)**:
  - Saves only the numeric weight matrices in HDF5 format.
- **Lines 91–97 (`accuracies.json`)**:
  - Exports training, validation, and test metrics to a human-readable JSON file for records, auditing, and visualization.
- **Lines 99–100 (`tokenizer.pkl`)**:
  - **Essential for production**: Serializes the fitted `Tokenizer` object. Without saving this, future prediction scripts would have no way to map words to the exact integer IDs learned during training.

---

# 3. Deep Dive: Natural Language Representation Concepts

### Why Can't We Feed Words Directly into Neural Networks?
Neural networks are mathematical functions consisting of matrix multiplications: $y = f(W \cdot x + b)$. They can only compute on numeric tensors. Words must be converted into numerical vectors.

### Evolution of Text Representations:

```
Raw Text: "great movie"
    │
    ▼
[Tokenization]        ──> Splitting text into tokens: ["great", "movie"]
    │
    ▼
[Integer Encoding]    ──> Replacing with vocabulary indices: [12, 17]
    │
    ▼
[One-Hot Encoding]    ──> High-dimensional sparse vectors (mostly 0s): [0, 0, ..., 1, ..., 0]
    │                     (Cons: Huge memory footprint, zero semantic relationship)
    ▼
[Dense Embeddings]    ──> Low-dimensional continuous vectors: [0.24, -0.81, 0.45, ...]
                          (Pros: Compact, captures rich semantic relationships)
```

### The Embedding Layer Explained
- An **Embedding Layer** is essentially a trainable lookup table:
  $$\text{Matrix Size} = \text{VOCAB\_SIZE} \times \text{EMBEDDING\_DIM} = 20,000 \times 64 = 1,280,000 \text{ parameters}$$
- When word index `12` enters the layer, it retrieves row 12 of the matrix: a dense 64-number vector.
- **Semantic Proximity**: During backpropagation, words that appear in similar emotional contexts (e.g. `"superb"`, `"wonderful"`, `"masterpiece"`) have their vectors pulled closer together in 64-dimensional geometric space. Words with opposing sentiments are pushed apart.

---

# 3. Deep Dive: Neural Network Architecture & The LSTM

### Why Not Traditional Machine Learning (e.g. Naive Bayes, Logistic Regression)?
Traditional models use **Bag-of-Words** or **TF-IDF**. They treat sentences as an unordered bag of words. They ignore:
1. **Word Order**: `"not good, terrible acting"` vs `"not terrible, good acting"` have identical word counts, but opposite meanings!
2. **Long-Distance Context**: A sentiment modifier at the start of a paragraph may determine the meaning of a clause at the end.

### Why Not a Standard Recurrent Neural Network (Vanilla RNN)?
A standard RNN tries to pass memory from one word to the next using a recurrent hidden state:
$$h_t = \tanh(W \cdot x_t + U \cdot h_{t-1} + b)$$

**The Vanishing Gradient Problem**:
When backpropagation unrolls over 200 time steps, the gradients must be repeatedly multiplied by the weight matrix $W$. If the weights are small ($< 1.0$), gradients shrink exponentially toward zero as they travel backward:
$$\lim_{k \to \infty} W^k = 0$$
As a result, the network cannot learn relationships between words separated by more than 5–10 steps.

---

### The Solution: LSTM (Long Short-Term Memory)

The LSTM solves the vanishing gradient problem using a dedicated highway called the **Cell State ($C_t$)**, regulated by three specialized gates:

```
                           Cell State Highway (C_{t-1} ───► C_t)
                               ┌─────────────────────────┐
                               │     x (Forget Gate)     │
                               │     + (Input Gate)      │
                               └────────────┬────────────┘
                                            │
                                            ▼
Input (x_t) & Prev State (h_{t-1}) ──► [3 Gates] ──► New Hidden State (h_t)
```

1. **Forget Gate ($f_t$)**: Decides what information to discard from the cell state.
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
   *(Outputs numbers between 0 and 1: 0 = completely forget, 1 = completely keep).*
2. **Input Gate ($i_t$) & Candidate State ($\tilde{C}_t$)**: Decides what new information to store in the cell state.
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Updating the Cell State ($C_t$)**: Combines the old memory and new memory:
   $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
   *(Because this operation is additive, gradients can flow backward through time without vanishing!)*
4. **Output Gate ($o_t$)**: Decides what parts of the cell state make it to the output hidden state $h_t$:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t * \tanh(C_t)$$

---

# 3. Deep Dive: Training Dynamics, Overfitting & Early Stopping

### What is Overfitting?
Overfitting occurs when a neural network has enough capacity to **memorize** idiosyncratic noise and specific sample quirks in the training dataset, rather than learning general linguistic patterns.

### How to Identify Overfitting (Examining Your Results)
In your training run without early stopping:
- **Epoch 1**: `loss: 0.3545 - acc: 84.74%` | `val_loss: 0.2843 - val_acc: 88.36%` (Healthy)
- **Epoch 2**: `loss: 0.1971 - acc: 92.81%` | `val_loss: 0.3147 - val_acc: 88.44%` (val_loss starts rising!)
- **Epoch 3**: `loss: 0.1378 - acc: 95.14%` | `val_loss: 0.3532 - val_acc: 88.23%` (Diverging)
- **Epoch 4**: `loss: 0.0955 - acc: 96.79%` | `val_loss: 0.4201 - val_acc: 88.00%` (Diverging further)
- **Epoch 5**: `loss: 0.0686 - acc: 97.77%` | `val_loss: 0.4714 - val_acc: 87.35%` (Severe Overfitting)

```
Loss
 │
 │      \                               /   <--- Validation Loss (Rises = Overfitting!)
 │       \_____________________________/
 │        \
 │         \_______________________________ <--- Training Loss (Continues falling)
 └─────────────────────────────────────────────
  0            Epoch 1 (Best)             Epoch 5
```

- While training loss dropped to `0.0686` and train accuracy reached `97.77%`, validation loss climbed from `0.2843` up to `0.4714`.
- **The Golden Rule**: The moment Validation Loss begins to rise while Training Loss continues to fall, the model is overfitting.

### How EarlyStopping Solves This
By adding:
```python
EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
```
- The callback tracks `val_loss` at each epoch.
- At Epoch 2, it observes that `val_loss` (`0.3147`) is worse than Epoch 1 (`0.2843`).
- Because `patience=1`, it immediately stops further training and halts Epochs 3, 4, and 5.
- Because `restore_best_weights=True`, it rolls the weights back to the exact checkpoint from **Epoch 1**, preserving optimal generalization performance!

---

# 3. Deep Dive: Metrics (Train vs. Validation vs. Test)

| Metric | What Data It Uses | When It Is Evaluated | Core Purpose |
| :--- | :--- | :--- | :--- |
| **Train Accuracy** | The training split (`X_train`) | Continuously during each epoch | Measures how well the network can fit the training data. |
| **Validation Accuracy** | The holdout split (`validation_split=0.1`) | At the end of each epoch | Used by the engineer/callback to tune hyperparameters and detect overfitting. Gradients are **never** calculated on this data. |
| **Test Accuracy** | Completely untouched split (`X_test`) | Evaluated once after all training is completed | Provides an honest, unbiased assessment of real-world generalization performance. |

---

# Section 4: Production Inference & Prediction Pipeline (`4-predict.py`)

This file is a lightweight, standalone production inference script. It contains no training code, requires no training dataset, and uses only the saved model artifacts to classify incoming text.

## 4. Line-by-Line Breakdown

### Lines 1–6: Minimal Inference Imports
```python
1: import os
2: os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
4: import pickle
5: import tensorflow as tf
6: from tensorflow.keras.preprocessing.sequence import pad_sequences
```
- Silences verbose C++ logs and imports only the minimal inference tools needed to run the saved model.
- Notice that `train_test_split`, `Adam`, `Sequential`, or `EarlyStopping` are **not** imported here because inference does not train or optimize weights.

---

### Lines 9–14: Loading Artifacts and Defining Constants
```python
9:  model = tf.keras.models.load_model("model.keras")
11: with open("tokenizer.pkl", "rb") as f:
12:     tokenizer = pickle.load(f)
14: MAX_LEN = 200
```
- **Line 9 (`load_model`)**: Reconstructs the exact neural network architecture and loads all trained parameter weights from `model.keras`.
- **Lines 11–12 (`pickle.load`)**: Restores the fitted `Tokenizer` with its complete 20,000-word vocabulary map.
- **Line 14 (`MAX_LEN = 200`)**: Must match the exact sequence length used during training so input tensors match the expected shape `(None, 200)`.

---

### Lines 16–20: The Prediction Function
```python
16: def predict(text):
17:     seq = tokenizer.texts_to_sequences([text])
18:     pad = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
19:     p = model.predict(pad, verbose=0)[0][0]
20:     return "positive" if p > 0.5 else "negative", float(p)
```
- **Line 17 (`tokenizer.texts_to_sequences([text])`)**:
  - Encodes the input string into a list of word indices using the loaded vocabulary.
  - Notice the input `[text]` is wrapped in a list: Keras expects a batch of inputs, even when predicting on a single review.
- **Line 18 (`pad_sequences(...)`)**:
  - Ensures the sequence length is exactly 200 using identical `pre` padding and truncation rules.
- **Line 19 (`model.predict(pad, verbose=0)`)**:
  - Executes a forward pass through the network: Embedding $\to$ LSTM $\to$ Dense $\to$ Sigmoid.
  - `verbose=0`: Suppresses the progress bar during inference for clean output.
  - `[0][0]`: Extracts the raw probability scalar from the 2D output tensor of shape `(1, 1)`.
- **Line 20**:
  - Thresholds the probability: if $p > 0.5$, the review is classified as `"positive"`; otherwise `"negative"`.
  - Returns both the label and the confidence probability.

---

### Lines 22–23: Example Predictions
```python
22: print(predict("this movie was great and the acting was wonderful"))
23: print(predict("terrible film not worth watching"))
```
- Output:
  ```python
  ('positive', 0.9602)  # 96.0% confident positive
  ('negative', 0.1586)  # 84.1% confident negative (1.0 - 0.1586)
  ```

---

# Quick Reference Glossary

- **Corpus**: A collection of written texts used for linguistic analysis.
- **Token**: An individual atomic segment of text (usually a single word).
- **Stopwords**: High-frequency grammatical words (e.g. "the", "and") carrying minimal topic or sentiment meaning.
- **Lemmatization**: Reducing words to their base dictionary lemma using grammar and vocabulary rules.
- **Vocabulary Size (`VOCAB_SIZE`)**: The maximum number of distinct words indexed and recognized by the model.
- **Out-Of-Vocabulary (`<OOV>`)**: A catch-all token representing any word not in the model's vocabulary.
- **Embedding**: A dense, low-dimensional vector representation where semantically similar words are close in vector space.
- **Sequence Padding**: Adding zeros to make text sequences uniform in length so they can be processed in matrix batches.
- **Cell State**: The long-term internal memory highway of an LSTM network.
- **Dropout**: A regularization technique that randomly zeros neuron activations during training to prevent memorization.
- **Binary Cross-Entropy**: A loss function that measures the divergence between true binary labels and predicted probabilities.
- **Early Stopping**: A mechanism to halt training automatically as soon as validation loss begins to rise, preventing overfitting.
- **Data Leakage**: Mistakenly exposing validation or testing data to the model during training or feature preparation, leading to unrealistically optimistic results.
