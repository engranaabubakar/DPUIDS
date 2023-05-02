#Deep leaning and machine learning methods implemented
# Import required libraries
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
#from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense,GRU,Embedding,Dropout,Flatten,Conv1D,MaxPooling1D,LSTM


#Deep Learning Models
#input_size
# -> CIC-DDoS2019 80
# -> CIC-IDS2018 76

def GRU_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(GRU(32, input_shape=(input_size,1), return_sequences=False)) #
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.build()
    print(model.summary())
    
    return model
###CNN
def CNN_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=(input_size,1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=32, kernel_size=16, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    
    return model
#LSTM
def LSTM_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(LSTM(32,input_shape=(input_size,1), return_sequences=False))
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    
    return model
#DNN
def DNN_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(Dense(2, activation='relu', input_shape=(input_size,)))
    #model.add(Dense(100, activation='relu'))   
    #model.add(Dense(40, activation='relu'))
    #model.add(Dense(10, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    
    return model


#SVM
#from sklearn.svm import SVC
#def SVM():
#    return SVC(kernel='linear')

#LRLogistic Regression (LR)

#def LR():
    #return LogisticRegression()
#from sklearn.linear_model import LogisticRegression
#import numpy as np

# define the logistic regression model
def LR():
    #return LogisticRegression(solver='lbfgs')
    return LogisticRegression()
#Gradient Descent (GD) 
def GD():
    return SGDClassifier()
def kNN():
    return KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
#Auxiliar Functions
#Train and test and spilt samples
def train_test(samples):
    # Import `train_test_split` from `sklearn.model_selection`
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Specify the data 
    X=samples.iloc[:,0:(samples.shape[1]-1)]
    
    # Specify the target labels and flatten the array
    #y= np.ravel(amostras.type)
    y= samples.iloc[:,-1]
    
    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test
# normalize input data

def normalize_data(X_train,X_test):
    # Import `StandardScaler` from `sklearn.preprocessing`
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    
    # Define the scaler 
    #scaler = StandardScaler().fit(X_train)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    
    # Scale the train set
    X_train = scaler.transform(X_train)
    
    # Scale the test set
    X_test = scaler.transform(X_test)
    
    return X_train, X_test
# Reshape data input

def format_3d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1], 1))

def format_2d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1]))


#deep = False for scikit-learn ML methods
# compile and train learning model

def compile_train(model,X_train,y_train,deep=True):
    
    if(deep==True):
        import matplotlib.pyplot as plt


        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        history = model.fit(X_train, y_train,epochs=2, batch_size=256, verbose=1)
        #model.fit(X_train, y_train,epochs=3)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        print(model.metrics_names)
    
    else:
        model.fit(X_train, y_train) #SVM, LR, GD
    
    print('Model Compiled and Trained')
    return model

    ##### testes(model,X_test,y_test,y_pred, deep=True) 
 #   Testing performance outcomes of the methods

#deep = False for scikit-learn ML methods

    # Testing performance outcomes of the methods

def testes(model,X_test,y_test,y_pred, deep=True):
    if(deep==False): 
        score = model.evaluate(X_test, y_test,verbose=1)

        print(score)
    
    # Alguns testes adicionais
    #y_test = formatar2d(y_test)
    #y_pred = formatar2d(y_pred)
    
    
    # Import the modules from `sklearn.metrics`
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score
    
    # Accuracy 
    acc = accuracy_score(y_test, y_pred)
    print('\nAccuracy')
    print(acc)
    
    # Precision 
    prec = precision_score(y_test, y_pred)#,average='macro')
    print('\nPrecision')
    print(prec)
    
    # Recall
    rec = recall_score(y_test, y_pred) #,average='macro')
    print('\nRecall')
    print(rec)
    
    # F1 score
    f1 = f1_score(y_test,y_pred) #,average='macro')
    print('\nF1 Score')
    print(f1)
    
    #average
    avrg = (acc+prec+rec+f1)/4
    print('\nAverage (acc, prec, rec, f1)')
    print(avrg)

    #TPR FPR NPR
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tpr = tp / (tp + fn)
    print('\n TNR')
    print(tnr)
    print('\n FPR')
    print(fpr)
    print('\n FNR')
    print(fnr)
    print('\n TPR')
    print(tpr) 
    
    return acc, prec, rec, f1, avrg, tnr, fpr, fnr, tpr

    #######3 test_normal_atk(y_test,y_pred): Calculate the correct classification rate of normal and attack flow records
""""
def test_normal_atk(y_test,y_pred):
    df = pd.DataFrame()
    df['y_test'] = y_test
    df['y_pred'] = y_pred
    normal = len(df.query('y_test == 0'))
    atk = len(y_test)-normal    
    wrong = df.query('y_test != y_pred')
    normal_detect_rate = (normal - wrong.groupby('y_test').count().iloc[0][0]) / normal
    atk_detect_rate = (atk - wrong.groupby('y_test').count().iloc[1][0]) / atk
    #print(normal_detect_rate,atk_detect_rate)
    return normal_detect_rate, atk_detect_rate"""

def test_normal_atk(y_test,y_pred):
    df = pd.DataFrame()
    df['y_test'] = y_test
    df['y_pred'] = y_pred
    normal = len(df.query('y_test == 0'))
    atk = len(y_test)-normal    
    wrong = df.query('y_test != y_pred')
    normal_detect_rate = 0
    atk_detect_rate = 0
    
    # Check if normal class is present in wrong predictions
    if 0 in wrong['y_test'].unique():
        normal_detect_rate = (normal - wrong.groupby('y_test').count().loc[0].get('y_pred', 0)) / normal
    
    # Check if attack class is present in wrong predictions
    if 1 in wrong['y_test'].unique():
        atk_detect_rate = (atk - wrong.groupby('y_test').count().loc[1].get('y_pred', 0)) / atk
    
    return normal_detect_rate, atk_detect_rate    

    #### Save Model

    # Save model and weights

def save_model(model,name):
    from keras.models import model_from_json
    
    arq_json = 'Model' + name + '.json'
    model_json = model.to_json()
    with open(arq_json,"w") as json_file:
        json_file.write(model_json)
    
    arq_h5 = 'Model' + name + '.h5'
    model.save_weights(arq_h5)
    print('Model Saved')
    
def load_model(name):
    from keras.models import model_from_json
    
    arq_json = 'Model/' + name + '.json'
    json_file = open(arq_json,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    arq_h5 = 'Model/' + name + '.h5'
    loaded_model.load_weights(arq_h5)
    
    print('Model loaded')
    
    return loaded_model

def save_Sklearn(model,nome):
    import pickle
    arquivo = 'Model'+ nome + '.pkl'
    with open(arquivo,'wb') as file:
        pickle.dump(model,file)
    print('Model sklearn saved')

def load_Sklearn(nome):
    import pickle
    arquivo = 'Model/'+ nome + '.pkl'
    with open(arquivo,'rb') as file:
        model = pickle.load(file)
    print('Model sklearn loaded')
    return model
######Dataset - CIC-DDoS2019

#Loading training dataset (day 1), upsampling normal flows for balancing the training set.

# UPSAMPLE OF NORMAL FLOWS
    
samples = pd.read_csv('CICDDoS2019/01-12-c/export_dataframe_proc.csv', sep=',')

X_train, X_test, y_train, y_test = train_test(samples)


#junta novamente pra aumentar o numero de normais
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
is_benign = X[' Label']==0 #base de dados toda junta

normal = X[is_benign]
ddos = X[~is_benign]

# upsample minority
normal_upsampled = resample(normal,
                          replace=True, # sample with replacement
                          n_samples=len(ddos), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([normal_upsampled, ddos])

# Specify the data 
X_train=upsampled.iloc[:,0:(upsampled.shape[1]-1)]    #DDoS
y_train= upsampled.iloc[:,-1]  #DDoS

input_size = (X_train.shape[1], 1)

del X, normal_upsampled, ddos, upsampled, normal #, l1, l2 

##### Day2 processing

tests = pd.read_csv('CICDDoS2019/01-12-c/export_tests_proc.csv', sep=',')

# X_test = np.concatenate((X_test,(tests.iloc[:,0:(tests.shape[1]-1)]).to_numpy())) # testar 33% + dia de testes
# y_test = np.concatenate((y_test,tests.iloc[:,-1]))

del X_test,y_test                            # testar só o dia de testes
X_test = tests.iloc[:,0:(tests.shape[1]-1)]                        
y_test = tests.iloc[:,-1]

# print((y_test.shape))
# print((X_test.shape))

X_train, X_test = normalize_data(X_train,X_test)



######## Model save and laod 
####Compiling and Training the methods

####Comment the last 2 code blocks

##OR

####Loading and compiling the methods

###Comment the first 2 code blocks

"""
model_gru = GRU_model(input_size=80) #quando treina novo modelo
model_cnn = CNN_model(input_size=80)
model_lstm = LSTM_model(input_size=80)
model_dnn = DNN_model(X_train.shape[1])
model_svm = SVM()
model_lr = LR()
model_gd = GD()
model_knn = kNN()


model_gru = compile_train(model_gru,format_3d(X_train),y_train)  #quando treina novo modelo, ou retreina
model_cnn = compile_train(model_cnn,format_3d(X_train),y_train)
model_lstm = compile_train(model_lstm,format_3d(X_train),y_train)
model_dnn = compile_train(model_dnn,X_train,y_train)
model_svm = compile_train(model_svm,X_train,y_train,False)
model_lr = compile_train(model_lr,X_train,y_train,False)
model_gd = compile_train(model_gd,X_train,y_train,False)
model_knn = compile_train(model_knn,X_train,y_train,False)

## Comment next 2 blocks if training new models
## Execute them if loading pre-trained models
"""
model_gru = load_model('GRU20-32-b256') #when loading previously saved trained model and weights
model_cnn = load_model('CNN5')
#model_lstm = load_model('LSTM5-32-b256')
#model_dnn = load_model('DNN5-2560')
#model_svm = SVM()
#model_svm = load_Sklearn('SVM') 
#model_lr = load_Sklearn('LR')
#model_gd = load_Sklearn('GD')
#model_knn = load_Sklearn('kNN')
#model_svm = SVM()
model_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #qdo carrega modelo salvo
model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model_dnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Testing CIC-DDoS2019
results = pd.DataFrame(columns=['Method','Accuracy','Precision','Recall', 'F1_Score', 'Average','Normal_Detect_Rate','Atk_Detect_Rate'])

print("GRU Results")

y_pred = model_gru.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc, prec, rec, f1, avrg, tnr, tpr, fnr, fpr = testes(model_gru,format_3d(X_test),y_test,y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

norm, atk = test_normal_atk(y_test,y_pred)
results_df = pd.DataFrame({
    'Method': ['GRU'],
    'Accuracy': [acc],
    'Precision': [prec],
    'F1_Score': [f1],
    'Recall': [rec],
    'Average': [avrg],
    'Normal_Detect_Rate': [norm],
    'Atk_Detect_Rate': [atk],
    'TNR': [tnr],
    'TPR': [tpr],
    'FNR': [fnr],
    'FPR': [fpr]
})

results = pd.concat([results, results_df], ignore_index=True)

#CNN
import subprocess
import pandas as pd

# Define functions for running CNN model and computing performance metrics
# ...

# Run CNN model and compute performance metrics
print("CNN Results")
y_pred = model_cnn.predict(format_3d(X_test)) 

y_pred = y_pred.round()
acc, prec, rec, f1, avrg, tnr, tpr, fnr, fpr = testes(model_cnn,format_3d(X_test),y_test,y_pred) 

norm, atk = test_normal_atk(y_test,y_pred)
results_df = pd.DataFrame({
    'Method': ['CNN'],
    'Accuracy': [acc],
    'Precision': [prec],
    'F1_Score': [f1],
    'Recall': [rec],
    'Average': [avrg],
    'Normal_Detect_Rate': [norm],
    'Atk_Detect_Rate': [atk],
    'TNR': [tnr],
    'TPR': [tpr],
    'FNR': [fnr],
    'FPR': [fpr]
})

results = pd.concat([results, results_df], ignore_index=True)

# Detect DDoS attack and execute terminal command if an attack is detected
if acc > 0.99:
    result = subprocess.run(['/opt/mellanox/doca/applications/ips/bin/doca_ips', '-a', '0000:03:00.0,class=regex', '-a', 'auxiliary:mlx5_core.sf.4,sft_en=1', '-a', 'auxiliary:mlx5_core.sf.5,sft_en=1', '-l', '0-7', '--', '--cdo', '/tmp/ddos_ids.cdo', '-p'], capture_output=True, text=True)
    print(result.stdout)

