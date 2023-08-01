import string

import nltk

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re


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
    
    if not nltk.corpus.exists('stopwords'):
        nltk.download('stopwords')
    
    review = self.to_lowercase(review)
    review = self.remove_stopwords(review)
    review = self.delete_punctuation(review)
    review = self.delete_numbers(review)
    review = self.stemming(review)

    return review