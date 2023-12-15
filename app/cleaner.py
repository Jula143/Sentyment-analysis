
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

import pkg_resources


class Cleaner:

  def delete_punctuation(self,review):
    
    review = review.translate(str.maketrans('', '', string.punctuation))
    return review

  def delete_numbers(self,review):
    
    new_review=re.sub("[0-9]"," ",review)
    return new_review

  def to_lowercase(self,review):
    
    review=review.lower()
    return review

  def remove_stopwords(self,review):
    
    new_review = " "
    words = review.split()
    new_review=new_review.join([word for word in words if word not in stopwords.words('english')])
    return new_review

  def stemming(self,review):
    
    words = review.split()
    stemmer = SnowballStemmer("english")
    new_review=" "
    new_review=new_review.join([stemmer.stem(word) for word in words])
    return new_review
  
  def process_review(self, review):
    
    review = self.to_lowercase(review)
    review = self.remove_stopwords(review)
    review = self.delete_punctuation(review)
    review = self.delete_numbers(review)
    review = self.stemming(review)

    return review

    
def check_sentiment(text):
    file_path = pkg_resources.resource_filename(__name__, '/model_logreg.pkl')
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    file_path = pkg_resources.resource_filename(__name__, '/vectorizer.pkl')
    with open(file_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    cl = Cleaner()    
    preprocessed_review = cl.process_review(text)
    vectorized_review = vectorizer.transform([preprocessed_review])
    prediction = model.predict(vectorized_review)
    if prediction[0] == '2':
        return "positive"
    else:
        return "negative"