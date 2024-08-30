#%% md
# <h1>Actividad Clasificador de topicos</h1>
# <h2>Martes 10 de Agosto</h2>
# <ol>
# <li>Elegir dos topicos</li>
# <li>Obtener al menos 50 frases para cada topico</li>
# <li>Definir un set con almenos 25 frases de cada tópico</li>
# <li>Definir el código para entrenar un modelo de SGD o Naive Bayes</li>
# </ol>
# 
# 
#%%
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from collections import defaultdict
import csv, sys
#%%
canciones_df=pd.read_csv('data/Canciones Romanticas.csv')
documentacion_df=pd.read_csv('data/Lenguajes de programacion.csv')
canciones_df=canciones_df.iloc[:, 0].tolist()
documentacion_df=documentacion_df.iloc[:, 0].tolist()

canciones_df
documentacion_df
#%%
canciones_df = list(map(str.lower,canciones_df))
documentacion_df = list(map(str.lower,documentacion_df))
#%%
canciones_df
#%%
categories = ['canciones', 'documentacion']
#%%
# Join individual documents to calculate cosine similarity
canciones = ' '.join(canciones_df)
documentacion = ' '.join(documentacion_df)
documents = [canciones,documentacion]
count_vectorizer = CountVectorizer(stop_words="english")
#count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)

doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(
   doc_term_matrix,
   columns=count_vectorizer.get_feature_names_out(),
   index=["canciones", "documentacion"],
)
print(df)
print(cosine_similarity(df, df))
#%%
df.shape
#%% md
# Identify common tokens and resolve
# 
# the first approach, delete all, is not right; delete those elements that are less frequent. Example: paper [105 0 5] delete for segment 15 and 32, if the number is the same, delete the word is all bags of words.
# 
# There are 201 terms.
# 
# Create three lists to keep track of which tokens will be erased for each segment
#%%
i = 0
commonTerms = []
cancionesdelete = []
documentaciondelete = []
for (columnName, columnData) in df.items():
    nOcurrences = sum(x != 0 for x in columnData.values)
    if nOcurrences > 1:
      i = i + 1
      commonTerms.append(columnName) # keep track of all the common terms
      print('Token : ', columnName)
      print('Token Frequency : ', columnData.values)
      print('Token Repetition', sum(x != 0 for x in columnData.values))
      cancionescount = columnData.values[0]
      documentacioncount = columnData.values[1]
      print('Canciones Token count ', cancionescount)
      print('Documentacion Token count ', documentacioncount)
      # Less frequent tokens are deleted from the bag of words
      if (cancionescount > documentacioncount) and (cancionescount > documentacioncount):
        print('Canciones wins')
        documentaciondelete.append(columnName)
      elif (documentacioncount > cancionescount) and (documentacioncount > cancionescount):
        print('Documentacion wins')
        cancionesdelete.append(columnName)
      else: # there is a tie, the tokens is destroyed
        cancionesdelete.append(columnName)
        documentaciondelete.append(columnName)
print('Total terms ', i)
#%%
cancionesc = []
for string in canciones_df:
  print(string)
  cancionesc.append(' '.join(i for i in string.split() if i not in cancionesdelete))
#%%
documentacionc = []
for string in documentacion_df:
  print(string)
  cancionesc.append(' '.join(i for i in string.split() if i not in documentaciondelete))
#%%
# Join individual documents to calculate cosine similarity after cleansing
# Reduction of the cosine similarity
canciones = ' '.join(cancionesc)
documentacion = ' '.join(documentacionc)
documents = [canciones, documentacion]
count_vectorizer = CountVectorizer(stop_words="english")
#count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)

doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(
   doc_term_matrix,
   columns=count_vectorizer.get_feature_names_out(),
   index=["canciones", "documentacion"],
)
print(df)
print(cosine_similarity(df, df))
#%%
fullTrainingDatac = cancionesc + documentacionc
cancionesvaluesc = [1] * len(cancionesc)
documentacionvaluesc = [2] * len(documentacionc)
fullDataClassificationc = cancionesvaluesc + documentacionvaluesc

count_vect = CountVectorizer(stop_words='english')
X_products_countsc = count_vect.fit_transform(fullTrainingDatac)
X_products_countsc.shape
#%%
fullTrainingData = canciones_df + documentacion_df
#%%
cancionesvalues = [1] * len(canciones_df)
documentacionvalues = [2] * len(documentacion_df)
fullDataClassification = cancionesvalues + documentacionvalues
#%%
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')
X_products_counts = count_vect.fit_transform(fullTrainingData)
X_products_counts.shape
#%% md
# These are the term matrices
#%%
doc_term_matrix = X_products_counts.todense()
df = pd.DataFrame(
   doc_term_matrix,
   columns=count_vect.get_feature_names_out()
)
print(df)
print(cosine_similarity(df, df))
#%% md
# Create Inverse Matrix
#%% md
# CountVectorizer team 2
#%%
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_products_counts)
X_train_tfidf.shape
#%%
classlookup = None
filename = 'data/test_data_ml_classifier(1).csv'
with open(filename, newline='') as f:
                reader = csv.reader(f)
                try:
                    classlookup = defaultdict(list)
                    for key, val in reader:
                        classlookup[key] = val
                except csv.Error as e:
                    sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
#%%
test_set_golden = list(classlookup.values())
#%%
test_set_golden = test_set_golden[1:]
#%%
test_set_golden = [int(i) for i in test_set_golden]
#%%
docs_test = list(classlookup.keys())[1:]
#%%
MultinomialNB
#%%
CountVectorizer
#%%
TfidfTransformer
#%%
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
#%%
text_clf.fit(fullTrainingData, fullDataClassification)
#%%
predicted = text_clf.predict(docs_test)
len(predicted)
len(test_set_golden)
#%%
np.mean(predicted == test_set_golden) # The Naive Bayes classifier has a precision of 79%
#%%
print(metrics.classification_report(test_set_golden, predicted,
                                    target_names=categories))
#%%
metrics.confusion_matrix(test_set_golden, predicted)
#%%
values, counts = np.unique(test_set_golden, return_counts=True)
#%%
values, counts = np.unique(predicted, return_counts=True)
#%% md
# Try with SVM Support Vector Machine
# 
# IMPORTANT: works better with individual documents rather than a single one; but cleaning will use the whole documents.
#%%
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(fullTrainingData, fullDataClassification)
predicted = text_clf.predict(docs_test)
np.mean(predicted == test_set_golden)
#%%
print(metrics.classification_report(test_set_golden, predicted,
                                    target_names=categories))
#%%
metrics.confusion_matrix(test_set_golden, predicted)