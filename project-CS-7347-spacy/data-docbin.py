""" 
EDA and text data visualization resources:
- https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
- https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/
- https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools

"""

"""
-> import relevant libraries
"""
# %%
import os
import pandas as pd
import numpy as np

# Spacy imports
import spacy
from spacy.lang.en import English
from spacy import displacy
from spacy.tokens import DocBin

# tqdm shows a progress bar while executing cells
from tqdm.auto import tqdm

# ML 
from sklearn.model_selection import train_test_split

"""
Load in data
"""
# %%
os.chdir("/Users/jeveragar/Documents/Developer/PythonFiles/Fall_2021/CS_7347/Project/project-native")
# %%
data_path = r'data/IMDB_Dataset.csv'
data = pd.read_csv(data_path)
# data.value_counts(subset='sentiment')

# %%
data_sample = data.sample(frac=.1, random_state=1)
# %%
data_sample.value_counts(subset='sentiment')
# %%
# split data into train, valid, and test datasets
X_train, X_rem, y_train, y_rem = train_test_split(data_sample['review'], data_sample['sentiment'], train_size=.6)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, train_size=.5)

d = {'review': X_train, 'sentiment': y_train}
train_df = pd.DataFrame(data=d)
del d

d = {'review': X_valid, 'sentiment': y_valid}
valid_df = pd.DataFrame(data=d)
del d

d = {'review': X_test, 'sentiment': y_test}
test_df = pd.DataFrame(data=d)
del d

# %%
# confirm balance of categories for each dataframe
print(train_df.shape)
print(valid_df.shape)
print(test_df.shape)

print(train_df.value_counts(subset='sentiment'))
print(valid_df.value_counts(subset='sentiment'))
print(test_df.value_counts(subset='sentiment'))

# %%
test_df.to_csv("data/test_data.csv", index=False)


"""
This is where the spacy stuff begins
"""
# %%
def make_docs(data):
    docs = []

    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total = len(data)):
        if label =="negative":
            doc.cats["positive"] = 0
            doc.cats["negative"] = 1
        else:
            doc.cats["positive"] = 1
            doc.cats["negative"] = 0
        docs.append(doc)

    return docs

# %%
nlp = spacy.load("en_core_web_sm")

# %%
train_docs = make_docs(train_df)
doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk("./data/train.spacy")

valid_docs = make_docs(valid_df)
doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk("./data/valid.spacy") # ./ gets you in the current working directory


# %%
#nlp.analyze_pipes()

# %%
