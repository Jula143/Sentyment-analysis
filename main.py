from src.preprocessing.data import *
from src.vectorization.methods import *
from src.classifiers.models import *

class Application:
        
    def starting_menu(self):
        while True:
            print("------Menu------")
            print("1. Create new model")
            print("2. Load default model")
            print("3. Compare models")
            print("4. Exit")

            choice=input("Choose action: ")
            print("\n")
            try:
                if int(choice)<1 or int(choice)>4:
                    print("Invalid choice. Try again.\n")
                    continue
            except ValueError:
                print("Invalid choice. Try again.\n")
            else: 
                break
        return choice
    
    def choose_model(self):
        while True:
            print("------Available models------")
            print("1. Naive Bayes")
            print("2. Logistic regression")
            print("3. SVM")
            print("4. K-nearest neighbours")
            choice=input("Choose model: ")
            print("\n")
            try:
                if int(choice)<1 or int(choice)>4:
                    print("Invalid choice. Try again.\n")
                    continue
            except ValueError:
                print("Invalid choice. Try again.\n")
            else: 
                break
        return choice
    
    def choose_new_dataset(self):
        while(True):
            print("Do you want to change dataset sizes:")
            print("1. Yes")
            print("2. No")
            choice=input("Choice: ")
            print("\n")
            try:
                if int(choice)<1 or int(choice)>2:
                    print("Invalid choice. Try again.\n")
                    continue
            except ValueError:
                print("Invalid choice. Try again.\n")
            else: 
                break
        return choice
    
    def choose_parameters(self):
        while True:
            print("1. Default hyperparameters")
            print("2. Choose hyperparameters")
            print("3. Hyperparameters tuning")
            
            choice=input("Choice: ")
            print("\n")
            try:
                if int(choice)<1 or int(choice)>3:
                    print("Invalid choice. Try again.\n")
                    continue
            except ValueError:
                print("Invalid choice. Try again.\n")
            else: 
                break
        return choice
    
    def parameters(self,model_num,model):
        if model_num=='1':
            while True:
                try:
                    alpha=float(input("alpha (positive float): "))
                    if(alpha<=0):
                        raise ValueError
                except ValueError:
                    print("Value Error. Please try different value")
                else:
                    break
            model.set_parameters(alpha)
        elif model_num=='2':
            l1_ratio = None
            while True:
                penalty=input("penalty (l1/l2/elasticnet): ") 
                if penalty!='l1' and penalty!='l2' and penalty!='elasticnet':
                    print("Value Error. Please try different value")
                    continue
                if penalty=='elasticnet':
                    while True:
                        try:
                            l1_ratio=float(input("l1_ratio (float [0,1]): "))
                            if l1_ratio<0 or l1_ratio>1:
                                raise ValueError
                        except ValueError:
                            print("Value Error. Please try different value")
                        else:
                            break
                break
            model.set_parameters(penalty,l1_ratio)
        elif model_num=='3':
            while True:
                kernel=input("kernel (linear/rbf/poly/sigmoid): ")
                if kernel!='linear' and kernel!='rbf' and kernel!='poly' and kernel!='sigmoid':
                    print("Value Error. Please try different value")
                    continue
                break
            model.set_parameters(kernel)
        elif model_num=='4':
            while True:
                try:
                    n = int(input("n_neighbours (int value above 1 and below your train size): "))
                    if n<1 or n>len(model.train_labels):
                        raise ValueError
                except ValueError:
                    print("Value Error. Please try different value")
                else:
                    break
            model.set_parameters(n)
            
        return model
    
    
    def load_model(self):
        while True:
            print("------Load model------")
            print("1. Naive Bayes")
            print("2. Logistic regression")
            print("3. SVM")
            print("4. K-nearest neighbours")
            choice=input("Choice: ")
            print("\n")
            try:
                if int(choice)<1 or int(choice)>4:
                    print("Invalid choice. Try again.\n")
                    continue
            except ValueError:
                print("Invalid choice. Try again.\n")
            else: 
                break
        
        if choice=='1':
            file_path = pkg_resources.resource_filename(__name__, '/default-models/bayes')
            file_path_info = pkg_resources.resource_filename(__name__, '/default-models/bayes-info')
        elif choice=='2':
            file_path = pkg_resources.resource_filename(__name__, '/default-models/regression')
            file_path_info = pkg_resources.resource_filename(__name__, '/default-models/regression-info')
        elif choice=='3':
            file_path = pkg_resources.resource_filename(__name__, '/default-models/svm')
            file_path_info = pkg_resources.resource_filename(__name__, '/default-models/svm-info')
        elif choice=='4':
            file_path = pkg_resources.resource_filename(__name__, '/default-models/knn')
            file_path_info = pkg_resources.resource_filename(__name__, '/default-models/knn-info')

        with open(file_path, 'rb') as f:
            loaded_object = pickle.load(f)
        with open(file_path_info, 'rb') as f:   
            model_info=pickle.load(f)
            
        loaded_model = Model(model_info['vec_method'])
        loaded_model.model=loaded_object
        loaded_model.model_name=model_info['method']
        loaded_model.test_labels=model_info['test_labels']
        loaded_model.predicted_probab=model_info['test_data_prob'] 
        loaded_model.test_labels_pred=model_info['predicted_labels']
        loaded_model.train_labels=np.zeros(model_info['train_size'])
        return loaded_model
    
                    
    def create_model(self,model_num,parameters_choice,train,test,train_l,test_l, vec_method_name):
        if parameters_choice=='1':
            model=self.create_default_model(model_num,train,test,train_l,test_l,vec_method_name)
            print("\nPerforming sentyment analysis...")  
            model.sentyment_analysis()
            print("Sentyment analysis completed.")  
        elif parameters_choice=='2':
            model=self.create_default_model(model_num,train,test,train_l,test_l,vec_method_name)
            model=self.parameters(model_num,model)
            print("\nPerforming sentyment analysis...")  
            model.sentyment_analysis()
            print("Sentyment analysis completed.")  
        else:
            model=self.create_default_model(model_num,train,test,train_l,test_l,vec_method_name)
            print("\nPerforming grid search...") 
            model.grid_search()
            print("Grid search completed...") 
        
        model.get_accuracy()
        print("\n")
        
        return model
                       
    
    def create_default_model(self,model_num,train,test,train_l,test_l,vec_method_name):
        if model_num=='1':
            model = Bayes(vec_method_name, train,test,train_l,test_l)
        elif model_num=='2':
            model = LogRegression(vec_method_name, train,test,train_l,test_l)
        elif model_num=='3':
            model = SVM(vec_method_name, train,test,train_l,test_l)
        elif model_num=='4':
            model = KNeighbours(vec_method_name, train,test,train_l,test_l)
        return model
    
    def decide_data_size(self):
        while True:
            try:
                train_size = int(input("Choose train dataset size (at least 30): "))
                test_size = int(input("Choose test dataset size (at least 10): "))
                
                if train_size<30 or test_size<10:
                    raise ValueError
            except ValueError:
                print("Value Error. Please try different datazet size.")
            else:
                break
        return (train_size,test_size)
    
    def choose_vectorization(self):      
        while True:
            print("------Available methods------")
            print("1. Yes/No Word Vector")
            print("2. Count Vectorization")
            print("3. TF-IDF")
            print("4. Word2vec")
            print("5. FastText")
            choice=input("Choose method: ")
            print("\n")
            try:
                if int(choice)<1 or int(choice)>5:
                    print("Invalid choice. Try again.\n")
                    continue
            except ValueError:
                print("Invalid choice. Try again.\n")
            else: 
                break
                
                
        return choice
        
    def vectorize_data(self,method_num,train,test):
        
        print("Data vectorization...")
        
        data = train.copy()
        data.extend(test)
        
        vectorization_method = ""
        if method_num=='1':
            vectorized_data = yes_no_vector(data)
            vectorization_method="Yes/No Vector"
        elif method_num=='2':
            vectorized_data = countVector(data)
            vectorization_method="Count Vectorizer"
        elif method_num=='3':
            vectorized_data = tfidf(data)
            vectorization_method="TF-IDF Vectorizer"
        elif method_num=='4':
            vectorized_data = word2vec(data)
            vectorization_method="Word2Vector"
        elif method_num=='5':
            vectorized_data = fasttext(data)
            vectorization_method="FastText"
            
        print("Data vectorization completed.\n")
            
        newTrain = vectorized_data[:len(train)] 
        newTest = vectorized_data[len(train):] 

        return newTrain, newTest, vectorization_method
        
    def show_roc_curve_comparison(self,models): #just for tabels
        while(True):
            print("\n1. Choose two models to compare ROC curves: ")
            print("2. Go back to main menu")
            
            choice=input("Choice: ")
            if choice=='1':
                nums=input("Choose two model numbers (ex. 1,2): ")
                nums=choice.split(',')  
                models[int(nums[0])-1].auc_roc_for_two(models[int(nums[1])-1],nums)
            if choice=='2':
                return      
             
            

def main():
    menu = Application()
    models = []
    
    train_size = 0
    test_size = 0
    train, train_l, test, test_l = [], [], [], []
         
    while True:
        choice = menu.starting_menu()
        if choice=='1':
                
            if train_size!=0 or train_size==0:
                c = 0
                if train_size!=0:
                    c = menu.choose_new_dataset()
                if c=='1' or train_size==0:
                    train_size, test_size = menu.decide_data_size()
                    train,train_l,test,test_l=load_data(train_size,test_size)
                    print("Preprocessing data...")
                    test=preprocess_data(test)
                    train=preprocess_data(train)
                    print("Preprocessing completed.\n")
                
            vec_method = menu.choose_vectorization()
            vec_train,vec_test, vec_method_name=menu.vectorize_data(vec_method,train,test)
            model_choice = menu.choose_model()
            choice_par=menu.choose_parameters()
            model = menu.create_model(model_choice,choice_par,vec_train,vec_test,train_l,test_l,vec_method_name)
            models.append(model)
                
        elif choice=='2':
                
            model=menu.load_model()
            models.append(model)
            print("Model loaded sucessfully\n")
                
        elif choice=='3':
            if len(models)==0:
                print("No models to compare\n")
                continue
                
            i = 1
            for m in models:
                print("-------------Model "+str(i)+"-----------------")
                m.printInfo()
                print("------------------------------------\n")
                i+=1
            
        else:
            quit()
                       
        
import gensim
if __name__=="__main__":
    print(gensim.__version__)
    main()
    

    
    