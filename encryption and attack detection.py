Code for Cryptography
Code for installing libraries
pip install cryptography
pip install pycryptodome
pip install scrypt
Code for importing libraries
from cryptography.fernet import Fernet
from Crypto.Cipher import AES
import scrypt, os, binascii
import numpy as np
Code for connecting to google drive
from google.colab import drive drive.mount('/content/drive')
Code for loading the file from google drive
File=np.loadtxt('/content/drive/MyDrive/Project/AESfile',delimiter=',',skiprows=0,dtype=str)
Code for creating files and writing content
with open('password', 'wb') as pw_file:
pw_file.write(Key)
with open('msg', 'wb') as msg_file:
msg_file.write(File)
Code for reading files and storing the data
with open('password', 'rb') as pw_file:
passwordFileContent = pw_file.read()
with open('msg', 'rb') as msg_file:
msgFileContent = msg_file.read()
Code for encryption
def encrypt_AES_GCM(msg, password):
kdfSalt = os.urandom(16)
secretKey = scrypt.hash(password, kdfSalt, N=16384, r=8, p=1, buflen=32)
aesCipher = AES.new(secretKey, AES.MODE_GCM)
ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
return (kdfSalt, ciphertext, aesCipher.nonce, authTag)
Code for decryption
def decrypt_AES_GCM(encryptedMsg, password):
(kdfSalt, ciphertext, nonce, authTag) = encryptedMsg
secretKey = scrypt.hash(password, kdfSalt, N=16384, r=8, p=1, buflen=32)
aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
return plaintext
Code for assigning contents of the file and key
msg = msgFileContent
password = passwordFileContent
Code for encryption of file
encryptedMsg = encrypt_AES_GCM(msg, password)
print("encryptedMsg=", binascii.hexlify(encryptedMsg[1]))
Code for overwriting encrypted file
with open('/content/drive/MyDrive/Project/AES file','wb') as writefile:
writefile.write(binascii.hexlify(encryptedMsg[1]))
Code for decryption of file
decryptedMsg = decrypt_AES_GCM(encryptedMsg, password)
print("decryptedMsg", decryptedMsg)
Code for overwriting decrypted file
with open('/content/drive/MyDrive/Project/AES file','wb') as writefile:
writefile.write(decryptedMsg)
Code for Attack Detection Mechanism
Code for importing libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
Code for connecting to google drive
from google.colab import drive drive.mount('/content/drive')
Code for loading the file from google drive
df=pd.read_csv('/content/drive/MyDrive/Project/Share ADM-file.csv')
Code for dividing the dataset for training and testing
temp=df.drop('type',axis=1)
x=temp.drop('hash',axis=1)
y=df['type']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
Code for creating, training and testing the model
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
