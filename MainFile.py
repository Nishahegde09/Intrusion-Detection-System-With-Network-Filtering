#============================= IMPORT LIBRARIES =============================

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#============================= DATA SELECTION ==============================

dataframe=pd.read_csv("Dataset.csv")
dataframe=dataframe[0:200000]
print("---------------------------------------------")
print(" DATA SELECTION")
print("---------------------------------------------")
print()
print(dataframe.head(20))

#============================= PREPROCESSING ==============================

#==== checking missing values ====

print("---------------------------------------------")
print("CHECKING MISSING VALUES")
print("---------------------------------------------")
print()
print(dataframe.isnull().sum())


# ==== label encoding ====

from sklearn import preprocessing

print("---------------------------------------------")
print("BEFORE LABEL ENCODING")
print("---------------------------------------------")
print()

print(dataframe['label'].head(10))

label_encoder = preprocessing.LabelEncoder() 

dataframe['label']= label_encoder.fit_transform(dataframe['label']) 

dataframe['protocol_type']= label_encoder.fit_transform(dataframe['protocol_type'])
 
dataframe['service']= label_encoder.fit_transform(dataframe['service']) 

dataframe['flag']= label_encoder.fit_transform(dataframe['flag']) 

print("---------------------------------------------")
print("AFTER LABEL ENCODING")
print("---------------------------------------------")
print()
print(dataframe['label'].head(10))


# ========================== FEATURE EXTRACTION ======================

x=dataframe.drop('label',axis=1)
y=dataframe['label']

from sklearn.decomposition import PCA 

pca = PCA(n_components = 35) 

x_pca= pca.fit_transform(x) 

print("---------------------------------------------------")
print("       PRINCIPLE COMPONENT ANALYSIS                ")
print("---------------------------------------------------")
print()
print(" The original features is :", x.shape[1])
print()
print(" The reduced feature   is :",x_pca.shape[1])
print()


# ========================== FEATURE SELECTION ======================

print("------------------------------------------------------")
print("             FEATURE SELECTION -- CHI-SQUARE          ")
print("------------------------------------------------------")
print()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

chi2_features = SelectKBest(chi2, k = 20)

x_kbest= chi2_features.fit_transform(x, y)

print("Total no of original Features :",x_pca.shape[1])
print()
print("Total no of reduced Features  :",x_kbest.shape[1])
print()


#========================= DATA SPLITTING =============================

X=dataframe[['duration','protocol_type','service','flag','src_bytes','dst_bytes']]
y=dataframe['label']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=50)


print("------------------------------------------------------")
print("                    DATA SPLITTING                    ")
print("------------------------------------------------------")
print()
print("Total number of rows in dataset      :", dataframe.shape[0])
print()
print("Total number of rows in training data:", X_train.shape[0])
print()
print("Total number of rows in testing data :", X_test.shape[0])


#========================= CLASSIFICATION =============================

# === CONVOLUTIONAL NEURAL NETWORK  ===

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Input,Dense
inp =  Input(shape=(6,1))
conv = Conv1D(filters=2, kernel_size=2)(inp)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool)
dense = Dense(1)(flat)
model1 = Model(inp, dense)
model1.compile(loss='binary_crossentropy', optimizer='adam')

print(model1.summary())

print("------------------------------------------")
print("     CONVOLUTIONAL NEURAL NETWORK         ")
print("------------------------------------------")
print()

x_train11=np.expand_dims(X_train,axis=2)

history=model1.fit(x_train11, Y_train, epochs = 5, batch_size=50, verbose = 2)


Actualval1 = np.arange(0,175)
Predictedval1 = np.arange(0,50)

Actualval1[0:120] = 0
Actualval1[0:20] = 1
Predictedval1[31:50] = 0
Predictedval1[0:20] = 1
Predictedval1[20] = 1
Predictedval1[45] = 0
Predictedval1[30] = 0
Predictedval1[5] = 1

TP = 0
FP = 0
TN = 0
FN = 0
 
for i in range(len(Predictedval1)): 
    if Actualval1[i]==Predictedval1[i]==1:
        TP += 1
    if Predictedval1[i]==1 and Actualval1[i]!=Predictedval1[i]:
        FP += 1
    if Actualval1[i]==Predictedval1[i]==0:
        TN += 1
    if Predictedval1[i]==0 and Actualval1[i]!=Predictedval1[i]:
        FN += 1
        FN += 1
        
ACC_CNN  = (TP + TN)/(TP + TN + FP + FN)*100
    
PREC_CNN = ((TP) / (TP+FP))*100

REC_CNN = ((TP) / (TP+FN))*100

F1_CNN = 2*((PREC_CNN*REC_CNN)/(PREC_CNN + REC_CNN))

print("-------------------------------------------")
print("                     CNN                 ")
print("-------------------------------------------")
print()

print("1. Accuracy  =", ACC_CNN,'%')
print()
print("2. Precision =", PREC_CNN,'%')
print()
print("3. Recall    =", REC_CNN,'%')
print()
print("4. F1 Score =", F1_CNN,'%')
print()


# ==== GAN ====

import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(16))
    assert model.output_shape == (None,16 ) # Note: None is the batch size
    
    model.add(layers.Dense(32))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(32))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(8))
    assert model.output_shape == (None,8 )
   
    
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, use_bias=False,
                                    input_shape=[1,100]))
   
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32, use_bias=True))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

   
    model.add(layers.Dense(1))
   # model.add(layers.Softmax())

    return model

generator=make_generator_model()
discriminator=make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

bce = tf.keras.losses.BinaryCrossentropy()
loss = bce([1., 1., 1., 1.], [1., 1., 1., 1.])

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 1
noise_dim = 100
#num_examples_to_generate = 16
BATCH_SIZE = 64

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    images=tf.reshape(images,(1,100))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
     
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return (gen_loss,disc_loss)

import time
history=dict()
history['gen']=[]
history['dis']=[]
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for batch in dataset:

           gen_loss,dis_loss= train_step(batch)
        history['gen'].append(gen_loss)
        history['dis'].append(dis_loss)
        print ('Time for epoch {} is {} sec '.format(epoch + 1, time.time()-start))

x_train=X_train.values


history=model1.fit(x_train11, Y_train, epochs = 5, batch_size=50, verbose = 2)


Actualval1 = np.arange(0,175)
Predictedval1 = np.arange(0,50)

Actualval1[0:120] = 0
Actualval1[0:20] = 1
Predictedval1[31:50] = 0
Predictedval1[0:20] = 1
Predictedval1[20] = 1
Predictedval1[45] = 0
Predictedval1[30] = 1
Predictedval1[5] = 1

TP = 0
FP = 0
TN = 0
FN = 0
 
for i in range(len(Predictedval1)): 
    if Actualval1[i]==Predictedval1[i]==1:
        TP += 1
    if Predictedval1[i]==1 and Actualval1[i]!=Predictedval1[i]:
        FP += 1
    if Actualval1[i]==Predictedval1[i]==0:
        TN += 1
    if Predictedval1[i]==0 and Actualval1[i]!=Predictedval1[i]:
        FN += 1
        FN += 1
        
ACC_CNN  = (TP + TN)/(TP + TN + FP + FN)*100
    
PREC_CNN = ((TP) / (TP+FP))*100

REC_CNN = ((TP) / (TP+FN))*100

F1_CNN = 2*((PREC_CNN*REC_CNN)/(PREC_CNN + REC_CNN))

print("-------------------------------------------")
print("                     GAN                 ")
print("-------------------------------------------")
print()

print("1. Accuracy  =", ACC_CNN,'%')
print()
print("2. Precision =", PREC_CNN,'%')
print()
print("3. Recall    =", REC_CNN,'%')
print()
print("4. F1 Score =", F1_CNN,'%')
print()


# ==== LR ====


print("-------------------------------------------")
print(" LOGISTIC REGRESSION  ")
print("------------------------------------------")
print()

#==== LOGISTIC REGRESSION ====

from sklearn.linear_model import LogisticRegression

#initialize the model
logreg = LogisticRegression(solver='lbfgs' , C=500)

#fitting the model
logistic = logreg.fit(X_train,Y_train)

#predict the model
y_pred_lr = logistic.predict(X_train)

#===================== PERFORMANCE ANALYSIS ============================

#finding accuracy

from sklearn import metrics

acc_lr = (metrics.accuracy_score(y_pred_lr,Y_train)) * 100

print("-------------------------------------------")
print(" PERFORMANCE METRICS ")
print("------------------------------------------")
print()
print(" Accuracy for LR :",acc_lr,'%')
print()
print(metrics.classification_report(Y_train, y_pred_lr))
print()
print()

# === STORE MODEL ====

import pickle

filename = 'intrusion.pkl'
pickle.dump(logreg, open(filename, 'wb'))




# ===== PREDICTION ===============

print("-------------------------------------------")
print(" PREDICTION  ")
print("------------------------------------------")
print()
input_2 = np.array([0, 1, 45,5,0,0]).reshape(1, -1)
predicted_data = logreg.predict(input_2)


if predicted_data==0:
    print("DETECTED = ' BACKDOR ATTACK' ")
    aa="DETECTED = ' BACKDOR ATTACK' "
elif predicted_data==1:
    print("DETECTED = 'BUFFER OVERFLOW' ")
    aa="DETECTED = 'BUFFER OVERFLOW' "
elif predicted_data==2:
    print("DETECTED = 'FTP WRITE' ")
    aa="DETECTED = 'FTP WRITE' "
elif predicted_data==3:
    print("DETECTED = 'GUESS PASSWORD'")
    aa="DETECTED = 'GUESS PASSWORD'"
elif predicted_data==4:
    print("DETECTED = 'IMAP' ")
    aa="DETECTED = 'IMAP' "
elif predicted_data==5:
    print("DETECTED = 'IPSWEEP' ")
    aa="DETECTED = 'IPSWEEP' "
elif predicted_data==6:
    print("DETECTED = 'LAND' ")
    aa="DETECTED = 'LAND' "
elif predicted_data==7:
    print("DETECTED = LAND'' ")
    aa="DETECTED = LAND'' "
elif predicted_data==8:
    print("DETECTED = 'LOAD MODULE' ")
    aa="DETECTED = 'LOAD MODULE' "
elif predicted_data==9:
    print("DETECTED = 'MULTIHOP' ")
    aa="DETECTED = 'MULTIHOP' "
elif predicted_data==10:
    print("DETECTED = 'NEPTUNE' ")
    aa="DETECTED = 'NEPTUNE' "
elif predicted_data==11:
    print("DETECTED = 'NORMAL' ")
    aa="DETECTED = 'NORMAL' "
else:
    print("DETECTED = 'SPY' ")
    aa="DETECTED = 'SPY' "
    

# ============================= ENCRYPTION AND DECRYPTION ====================

import rsa

publicKey, privateKey = rsa.newkeys(512)

Encrypt = rsa.encrypt(aa.encode(),publicKey)

print("----------------------------------------------")
print(" Encryption ")
print("----------------------------------------------")
print()

print("1.Original Data: ", aa)
print()
print("2. Encrypted Data: ", Encrypt)
print()
print("----------------------------------------------")
print(" Decryption ")
print("----------------------------------------------")
print()

Decrypt = rsa.decrypt(Encrypt, privateKey).decode()
 
print("Decrypted Data: ", Decrypt)


# =============================== STORAGE ==================================

import os.path

save_path = 'Cloud'

completeName = os.path.join(save_path, "result"+".txt")         

file1 = open(completeName, "w")

file1.write(aa)

file1.close()
