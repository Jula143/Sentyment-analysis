import numpy as np

def yes_no_vector(reviews):
  
  
  #find unique words
  #unique_words = []
  #for d in reviews:
  #  d=d.split()
  #  for word in d:
  #    if len(word)<2:
   #     continue
   #   if word not in unique_words:
   #     unique_words.append(word) 
   
  #create vector for each review
  #vectors = []
  #for d in reviews:
  #  d=d.split()
  #  review_vector = np.zeros(len(unique_words),dtype=int)
  #  for word in d:
  #    if len(word)<2:
  #      continue
  #    index = unique_words.index(word)
  #    review_vector[index]=1
  #  vectors.append(review_vector)
    
  #return vectors

  vect=CountVectorizer(binary=True,max_features=10000,min_df=2)
  count_matrix=vect.fit_transform(reviews)
  
  return count_matrix


from sklearn.feature_extraction.text import CountVectorizer

def countVector(reviews): 
  vect=CountVectorizer(max_features=10000,min_df=2)
  count_matrix=vect.fit_transform(reviews)
  
  return count_matrix



from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(data):
    Tfidf_vect = TfidfVectorizer(max_features=10000,min_df=2)
    vectors =Tfidf_vect.fit_transform(data)
    
    
    return vectors

import gensim

def word_vector(tokens, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  
            continue
    if count != 0:
        vec /= count
    return vec

def word2vec(data): 
    tokenized_data=[]
    for i in range(len(data)):
        tokenized_data.append(data[i].split())

       
    model_w2v = gensim.models.Word2Vec(
                tokenized_data,
                vector_size=100, # desired no. of features/independent variables
                window=7, # context window size (7 worse)
                min_count=2, # Ignores all words with total frequency lower than 2.                                  
                sg = 1, # 1 for skip-gram model
    ) 

    model_w2v.train(tokenized_data, total_examples = len(tokenized_data), epochs=10)

    wordvec_arrays = np.zeros((len(tokenized_data), 100)) 
    for i in range(len(tokenized_data)):
        wordvec_arrays[i,:] = word_vector(tokenized_data[i], 100, model_w2v)
        
    return wordvec_arrays
        
        
from gensim.models.fasttext import FastText


def fasttext(data):
    tokenized_data=[]
    for i in range(len(data)):
        tokenized_data.append(data[i].split())
        
    model = FastText(vector_size=100, window=7, min_count=2, sg=1)  
    model.build_vocab(tokenized_data)
    model.train(tokenized_data, total_examples=len(data), epochs=10)  # train

    fasttext_arrays = np.zeros((len(tokenized_data), 100)) 
    for i in range(len(tokenized_data)):
        fasttext_arrays[i,:] = word_vector(tokenized_data[i], 100, model)
        
    return fasttext_arrays
  
  
from sklearn.preprocessing import MinMaxScaler 

def convert_to_positive(data):
  scaler = MinMaxScaler()
  pos = scaler.fit_transform(data)
  return pos