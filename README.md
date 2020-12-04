# My CNN implementation for the Fashion Mnist dataset for educational purposes

# Importing essential libraries
```python
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
```

# Reading data
```python
# importing data
train_df = pd.read_csv(r'C:\Users\asus\Desktop\fashion\fashion-mnist_train.csv', sep=',')
test_df = pd.read_csv(r'C:\Users\asus\Desktop\fashion\fashion-mnist_test.csv', sep=',')

""" 
You can download the dataset like me or use direct from keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()""";
```


```python
print(train_df.shape)
print(test_df.shape)
```

    >>>(60000, 785)
    >>>(10000, 785)
    


```python
# puting our data in np array
train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')
```


```python
# Visualization of the data
```

# Visualizing data
```python
w_grid = 5
l_grid = 10
# creating subplot area with 5 witdth and 10 lenght
# figsize is total plot area
fig, axes = plt.subplots(w_grid, l_grid, figsize=(16,10)) 
# reducing our array to one dimension, so we can iterate over it easily
axes = axes.reshape(-1)

for i in np.arange(0, w_grid * l_grid ):
    index = np.random.randint(0, len(train_data))
    axes[i].imshow(train_data[index,1:].reshape(28,28))
    axes[i].set_title(train_data[index, 0], fontsize=10)
    axes[i].axis('off')
    
#plt.subplots_adjust(hspace=0.2)
```


![png](https://github.com/Sekomer/CNN-Fashion/blob/main/photos/output_5_0.png)



```python
# split label and features then do generalization
X_train = train_data[:,1:]/255
y_train = train_data[:,0]

X_test = test_data[:,1:]/255
y_test = test_data[:,0]
```


```python
# to prevent overfit we use dev set
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```


```python
X_train.shape
# we will feed the convolutional network so we should change the dimensions
```




    >>>(48000, 784)




```python
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_dev = X_dev.reshape(X_dev.shape[0], 28, 28, 1)

print(X_train.shape, X_test.shape, X_dev.shape)
```

    >>>(48000, 28, 28, 1) (10000, 28, 28, 1) (12000, 28, 28, 1)
    

# Creating the model


```python
cnn_model = keras.Sequential()

cnn_model.add(Conv2D(64,3, 3, input_shape = (28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(32, activation = 'relu'))
cnn_model.add(Dense(10, activation = 'softmax'))
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.003),metrics =['accuracy'])
```


```python
%%capture
cnn_model.fit(X_train,
                        y_train,
                        batch_size = 256,
                        epochs= 50,
                        verbose = 1,
                        validation_data = (X_dev, y_dev))
```


```python
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy: ', evaluation[1])
```

    >>>313/313 [==============================] - 0s 895us/step - loss: 0.3147 - accuracy: 0.8830
    >>>Test Accuracy:  0.8830000162124634
    


```python
# model.predicted_classes() will be deprecated :p
predicted_classes = np.argmax(cnn_model.predict(X_test), axis=-1)
```


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (15,8))
sb.heatmap(cm, annot=True, fmt='d')
plt.xlabel('PREDICTION')
plt.ylabel('CLASS')

```




    >>>Text(114.0, 0.5, 'CLASS')




![png](https://github.com/Sekomer/CNN-Fashion/blob/main/photos/output_15_1.png)



```python
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(y_test, predicted_classes, target_names = target_names))
```

                    precision    recall  f1-score   support
    
         Class 0       0.79      0.88      0.83      1000
         Class 1       0.98      0.98      0.98      1000
         Class 2       0.87      0.78      0.82      1000
         Class 3       0.89      0.91      0.90      1000
         Class 4       0.77      0.84      0.81      1000
         Class 5       0.98      0.94      0.96      1000
         Class 6       0.71      0.62      0.66      1000
         Class 7       0.91      0.96      0.93      1000
         Class 8       0.97      0.97      0.97      1000
         Class 9       0.96      0.95      0.95      1000
    
        accuracy                           0.88     10000
       macro avg       0.88      0.88      0.88     10000
    weighted avg       0.88      0.88      0.88     10000
    
    
