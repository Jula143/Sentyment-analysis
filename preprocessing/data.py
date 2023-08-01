
from src.preprocessing.cleaner import Cleaner
import pkg_resources
from sklearn.model_selection import train_test_split

def load_data(train_size,test_size): 
    
    resource_package = __name__
    resouce_path = '/reviews.txt'
    with pkg_resources.resource_stream(resource_package,resouce_path) as f:
        file=f.read().decode().splitlines()

    test = []
    labels_test = []
    train = []
    labels_train = []
      
    i=0
    for line in file:  
        if i<train_size:
            train.append(line[10:])
            labels_train.append(line[9])
        else:
            test.append(line[10:])
            labels_test.append(line[9])
        if i==train_size+test_size-1:
            break
        i+=1
      
    return train,labels_train,test,labels_test



def preprocess_data(data):
    clean_text = Cleaner()
    for i in range(len(data)):
        data[i] = clean_text.process_review(data[i])
        
    return data
