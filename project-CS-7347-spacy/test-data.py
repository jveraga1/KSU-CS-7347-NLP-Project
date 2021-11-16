# %%
import spacy
import pandas as pd
# %%
test_data = pd.read_csv("data/test_data.csv")
# %%
test = test_data.values.tolist()
# %%
text = test[1]

# %%
nlp = spacy.load("output/model-last")

# %%
doc = nlp(text[0])

# %%
print(doc.cats)

# %%
print(text)
# %%
