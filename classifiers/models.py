from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
import pickle

class Model:
       
    def __init__ (self,vec_method):
        self.test_data=[]
        self.train_data=[]
        self.train_labels=[]
        self.test_labels=[]
        self.test_labels_pred=[]
        self.model=""
        self.model_name = ""
        self.vectorization_method=vec_method
        self.predicted_probab = []
        
    def set_data(self,train, test, train_l,test_l):
        self.test_data=test
        self.train_data=train
        self.train_labels=train_l
        self.test_labels=test_l

    def set_train_data(self,train,train_labels):
        self.train_data=train
        self.train_labels=train_labels

    def set_test_data(self,test,test_labels):
        self.test_data=test
        self.test_labels=test_labels
     
    def save_model(self,filename,model_name):
        if self.model!="":
            pickle.dump(self.model,open(filename,'wb'))
            model_info = {'method':model_name,'vec_method':self.vectorization_method,'train_size':len(self.train_labels),
                          'test_labels':self.test_labels,'test_data_prob':self.model.predict_proba(self.test_data)[:,1],'predicted_labels':self.test_labels_pred}
            pickle.dump(model_info,open(filename+'-info','wb'))          
    
    def get_accuracy(self):
        print('Accuracy:', round(accuracy_score(self.test_labels, self.test_labels_pred),3))

    def get_precision(self):
        print('Precision: ',round(metrics.precision_score(self.test_labels,self.test_labels_pred,pos_label='2'),3))

    def get_recall(self):
        print('Recall:', round(metrics.recall_score(self.test_labels, self.test_labels_pred,pos_label='2'),3))

    def get_f1score(self):
        print('F1 score:', round(f1_score(self.test_labels, self.test_labels_pred, average="macro"),3))

    def confusion_matrix(self):     
        cm = metrics.confusion_matrix(self.test_labels, self.test_labels_pred, labels=self.model.classes_)
        
        tn = cm[0, 0]  # True negatives
        fp = cm[0, 1]  # False positives
        specificity = tn / (tn + fp)
        print("Specificity: ",round(specificity,3))
        
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['negative','positive'])
        disp.plot()
        plt.show()
        
    def auc_roc(self):
        from sklearn.metrics import roc_curve, auc
        
        # Get predicted probabilities for positive class
        if len(self.predicted_probab)!=0:
            y_score=self.predicted_probab
        else:     
            y_score = self.model.predict_proba(self.test_data)[:,1]
        
        y_test = [int(i) for i in self.test_labels]

        # y_score are the true labels and y_score are the predicted probabilities
        fpr, tpr, _ = roc_curve(y_test, y_score,pos_label=2)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def printInfo(self):
        print("Model: "+self.model_name)
        print("Vectorization method: "+self.vectorization_method)
        print("Train data size: "+str(len(self.train_labels)))
        print("Test data size: "+str(len(self.test_labels)))
        self.get_accuracy()
        self.get_precision()
        self.get_recall()
        self.get_f1score()
        self.confusion_matrix()
        self.auc_roc()
       
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import GridSearchCV   
        
class Bayes(Model):
    
    def __init__(self, vec_method, train, test,train_l,test_l):       
        super().__init__(vec_method)
        self.set_data(train,test,train_l,test_l)
        self.alpha = 1.0
        self.model=MultinomialNB()
        
        
    def printInfo(self):
        self.model_name="Naive Bayes"
        super().printInfo()
        print("Alpha: ",self.alpha)
        
       
    def sentyment_analysis(self):  
        try:  
            self.model=self.model.fit(self.train_data, self.train_labels)
            self.test_labels_pred=self.model.predict(self.test_data)
        except ValueError:
            scaler = MinMaxScaler()
            self.train_data = scaler.fit_transform(self.train_data)
            self.test_data = scaler.transform(self.test_data)
            self.model=self.model.fit(self.train_data, self.train_labels)
            self.test_labels_pred=self.model.predict(self.test_data)
            
    def grid_search(self):
        parameters = {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}
        
        try:  
            grid_model = GridSearchCV(MultinomialNB(), parameters)
            grid_model.fit(self.train_data,self.train_labels)
            self.model = grid_model.best_estimator_
            self.test_labels_pred=self.model.predict(self.test_data)
        except ValueError:
            scaler = MinMaxScaler()
            self.train_data = scaler.fit_transform(self.train_data)
            self.test_data = scaler.transform(self.test_data)
            grid_model = GridSearchCV(MultinomialNB(), parameters)
            grid_model.fit(self.train_data,self.train_labels)
            self.model = grid_model.best_estimator_
            self.test_labels_pred=self.model.predict(self.test_data)

        self.alpha=grid_model.best_params_
                   
        
    def set_parameters(self,a): 
        self.alpha=a       
        self.model=MultinomialNB(alpha=a)  



from sklearn.linear_model import LogisticRegression

class LogRegression(Model):
    
    def __init__(self, vec_method, train, test,train_l,test_l):       
        super().__init__(vec_method)
        self.set_data(train,test,train_l,test_l)
        self.model=LogisticRegression(max_iter=10000,n_jobs=-1)
        self.penalty='l2'
        
    def printInfo(self):
        self.model_name="Logistic Regression"
        super().printInfo()
        print("Penalty: ",self.penalty)
        
    def sentyment_analysis(self):    
        self.model=self.model.fit(self.train_data, self.train_labels)
        self.test_labels_pred=self.model.predict(self.test_data)
        
    def grid_search(self):
        parameters = {"penalty": ['l1','l2',None]}
       # parameters = {"penalty": ['elasticnet'],
       #               "l1_ratio":[0.2,0.5,0.8]}
        grid_model = GridSearchCV(LogisticRegression(max_iter=10000,solver='saga',n_jobs=-1), parameters)
        grid_model.fit(self.train_data,self.train_labels)
        self.model = grid_model.best_estimator_
        self.test_labels_pred=self.model.predict(self.test_data)
        self.penalty=grid_model.best_params_

      
    def set_parameters(self,pen,l1_r):  
        self.penalty=pen      
        self.model=LogisticRegression(penalty=pen,solver='saga',l1_ratio=l1_r,max_iter=10000,n_jobs=-1)



from sklearn import svm
        
class SVM(Model):
    
    def __init__(self, vec_method, train, test,train_l,test_l):       
        super().__init__(vec_method)
        self.kernel='rbf'
        self.set_data(train,test,train_l,test_l)
        self.model=svm.SVC(probability=True)
        
    def printInfo(self):
        self.model_name="SVM"
        super().printInfo()
        print("Kernel: ",self.kernel)
        
    def grid_search(self):
        parameters = {"kernel": ['linear','rbf','poly','sigmoid']}

        grid_model = GridSearchCV(svm.SVC(probability=True), parameters)
        grid_model.fit(self.train_data,self.train_labels)
        self.model = grid_model.best_estimator_
        self.test_labels_pred=self.model.predict(self.test_data)
        self.kernel=grid_model.best_params_
        
    def set_parameters(self,ker):   
        self.kernel = ker     
        self.model=svm.SVC(kernel=ker,probability=True) 
                  
    def sentyment_analysis(self):
        
        self.model.fit(self.train_data,self.train_labels)
        self.test_labels_pred=self.model.predict(self.test_data)
   
   
from sklearn.neighbors import KNeighborsClassifier

class KNeighbours(Model):
    
    def __init__(self, vec_method, train, test,train_l,test_l):       
        super().__init__(vec_method)
        self.n_neighbours=3
        self.set_data(train,test,train_l,test_l)
        self.model=KNeighborsClassifier(n_jobs=-1)
        
    def printInfo(self):
        self.model_name="K-nearest Neighbours"
        super().printInfo()
        print("N neighbours: ",self.n_neighbours)
        
    def grid_search(self):
        parameters = {"n_neighbors": range(1, 11)}

        grid_model = GridSearchCV(KNeighborsClassifier(), parameters)
        grid_model.fit(self.train_data,self.train_labels)
        self.model = grid_model.best_estimator_
        self.test_labels_pred=self.model.predict(self.test_data)
        
        self.n_neighbours=grid_model.best_params_
              
    def sentyment_analysis(self):
        
        self.model.fit(self.train_data,self.train_labels)
        self.test_labels_pred=self.model.predict(self.test_data)
        
    def set_parameters(self,n):
        self.n_neighbours=n
        self.model=KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        
    
