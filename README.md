# Amazon review sentiment classification

A binary text classification project with separate training and test CSVs. Both files contain `Sentence` and `Polarity` columns. The training set contains 2,400 labeled sentences and the test set contains 600.

## Run the baseline

From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install scikit-learn
python train.py
```

[The baseline](train.py) fits a TF–IDF bigram and unigram vectorizer and logistic regression on the training CSV. It reports precision, recall and F1 on the test CSV. Use `python train.py --help` to see how to provide alternative input paths.
