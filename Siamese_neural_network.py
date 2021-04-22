'''
IFN680 ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING ASSIGNMENT 2

Developed by:
    - Huynh Ngoc Tram Nguyen (n10596283)
    - Filipe Arena (n10347682) 
    - Viet Thang Doan (n10301721)
 
A neural network can be trained to learn equivalence relations between objects. 
The core idea is to learn an embedding function such that equivalent objects are mapped to close points and non- equivalent objects are mapped to points that are far apart.
These neural networks are called Siamese neural networks because they are used in tandem on two different input vectors.
Applications of Siamese networks range from recognizing handwritten checks, automatic detection of faces in camera images, animal in the wild re-identification and matching queries with indexed documents.
In this assignment, we design a Siamese network to predict whether two glyphs belong to the same alphabet.

OMNIGLOT dataset contains 105x 105 character images in 50 different alphabets. 
There are 20 images of characters in each alphabet. 

In this assignment, we carry out the following tasks:
    - Load OMNIGLOT dataset and split the data
    - Implement and test the contrastive loss function 
    - Implement and test the triplet loss function 
    - Build a Siamese network 
    - Train your Siamese network on the training set
    - Evaluate the performance of your network trained with the different losses 
'''

#import statements
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Lambda, Input, Dropout, GlobalAveragePooling2D, BatchNormalization, concatenate
from keras.optimizers import Adadelta, Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from keras.regularizers import l2

#------------------------\
# I. GLOBAL VARIABLES     |
#------------------------/

#reshaped image dimension
reshaped_dimension=28

#model optimizer
rms = keras.optimizers.RMSprop()

#number of classes in OMNIGLOT dataset = 50 (as there are 50 different alphabets)
num_classes = 50

#global margin to seperate the distances between 2 images with different alphabets 
margin_contrastive_loss = 2
margin_triplet_loss = 0.5

# Define ranges with for different data sets based on the target label of the different pictures.
num_classes_1 = list(range(0,30)) #80% of this instances will be used for training, 20% will be used for testing set1
num_classes_2 = list(range(30,50)) #this is used for testing set2

# Stamp for opening Data Verification at the first execution
initiated = False

#------------------------\
# II. DEFINE FUNCTIONS    |
#------------------------/

#*****************************************************************************************************************
#*********************************************AUXILIARY FUNCTIONS*************************************************
#*****************************************************************************************************************

def resize_image(image,label,reshaped_dimension=28): 
    '''
    This function is to resize the image into smaller dimensions
    In this assignment, the image dimension will be rescale to 28x28 for faster computing
    '''
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image,(reshaped_dimension,reshaped_dimension))
    return image,label

def euclidean_distance(vects):
    '''
    This function is to compute the Euclidian Distance between two vectors in a Keras layer.
    Euclidean Distance is calculated based on the formula:
        A(A1,A2)    B(B1,B2) 
        Distance(A,B)=squareroot((A1-B1)^2 + (A2-B2)^2)
    @param:  vects - vectors (two input images)
    @return: the distance value
    '''
    x1, x2 = vects
    return K.sqrt(K.maximum(K.sum(K.square(x1 - x2), axis=1, keepdims=True), K.epsilon()))
  
def accuracy(y_true, y_pred):
    '''
    The function is to calculate the classification accuracy with a fixed threshold =0.5 on the distances in the tensor layer.
    @param:
      y_true: the target label -  equals 0 if the two images are similar and 1 otherwise
      y_predict: the Euclidean distance between the two image features 

    @return: accuracy value
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))



#****************************************************************************************************************************
#**********************LOAD, RESIZE IMAGES, SPLIT DATA INTO DIFFERENT SETS, AND NORMALIZIE DATA******************************
#****************************************************************************************************************************

def generate_data():
    # Load data online
    (ds_train, ds_test), ds_info = tfds.load(name='omniglot', split=['train', 'test'], with_info=True)
    
    #get the image and labels of training and test sets, and reshape images into 28x28 dimensions 
    train_labels = ([glyph["alphabet"] for glyph in ds_train])
    train_images = ([glyph["image"] for glyph in ds_train])
    train_images,train_labels=resize_image(train_images,train_labels)

    test_labels= ([glyph["alphabet"] for glyph in ds_test])
    test_images = ([glyph["image"] for glyph in ds_test])
    test_images,test_labels=resize_image(test_images,test_labels)

    #convert labels and images into numpy array
    train_labels= np.array(train_labels)
    train_images = np.array((train_images)[:, :, :,0])
    test_labels= np.array(test_labels)
    test_images = np.array((test_images)[:, :, :,0] )

    # Data combination
    full_images = np.concatenate((train_images,test_images))
    full_labels = np.concatenate((train_labels,test_labels))
        
    img_rows, img_cols = full_images.shape[1:3]
    input_shape = (img_rows, img_cols, 1)
        
    # Data Classification such that:
    #  keep 80% of images for training (num_classes_1)
    #  20% are used for one testing set (num_classes_2)

    images_set_1, labels_set_1 = [], []
    test_images_1, test_labels_1 = [], []
    
    for i in range(len(full_images)):
      if full_labels[i] in num_classes_1:
        images_set_1.append(full_images[i])
        labels_set_1.append(full_labels[i])
      elif full_labels[i] in num_classes_2: 
        test_images_1.append(full_images[i])
        test_labels_1.append(full_labels[i])
        
    # Convert the data type to np.array
    images_set_1 = np.array(images_set_1)
    labels_set_1 = np.array(labels_set_1)
    test_images_1 = np.array(test_images_1)
    test_labels_1 = np.array(test_labels_1)
    
    # Split the data into 8:2 for training & testing    
    train_images_1, test_images_2, train_labels_1, test_labels_2 = model_selection.train_test_split(images_set_1, labels_set_1, test_size = 0.2)
    
    # Combine the test set 1 and test set 2 into test set 3
    test_images_3 = np.concatenate((test_images_1, test_images_2))
    test_labels_3 = np.concatenate((test_labels_1, test_labels_2))

    # Data Optimisation
    # Reshape image arrays to 4D (batch_size, rows, columns, channels)
    train_images_1 = train_images_1.reshape(train_images_1.shape[0], img_rows, img_cols, 1) # 80%
    test_images_1 = test_images_1.reshape(test_images_1.shape[0], img_rows, img_cols, 1) # 100%
    test_images_2 = test_images_2.reshape(test_images_2.shape[0], img_rows, img_cols, 1) # 20%
    test_images_3 = test_images_3.reshape(test_images_3.shape[0], img_rows, img_cols, 1) # 100% + 20%
    
    # Data Normalization 
    # alter data type to float32
    train_images_1 = train_images_1.astype("float32")
    test_images_1 = test_images_1.astype("float32")
    test_images_2 = test_images_2.astype("float32")
    test_images_3 = test_images_3.astype("float32")
    #convert to float32 and rescale between 0 and 1
    train_images_1 /= 255
    test_images_1 /= 255
    test_images_2 /= 255
    test_images_3 /= 255

    # Create training data for 4 sets of data: training data set, testing set 1, testing set 2, testing set 3
    class_names = list(range(num_classes)) 

    return train_images_1, train_labels_1, test_images_1, test_labels_1, test_images_2, test_labels_2, test_images_3, test_labels_3, class_names, full_images, full_labels, images_set_1, labels_set_1, input_shape



#****************************************************************************************************************************
#************************************** CREATE PAIRS OF IMAGES AND  BUILD CONTRASIVE LOSS************************************
#****************************************************************************************************************************

def create_list_of_pairs(images, labels, digit_indices, num_classes):
    '''
    This function is to create positive and negative pairs of images . 
    @param:  
        + images: the input images
        + num_classes: the number of classes. The maximum number of classes is 50 if we would like to get all different classes involved.   
        + digit_indices: include the indexes of images in each class classified, for example:
            class 0 -> [[Image1],[Image2] ... [Image20]]
            class 1 -> [[Image1],[Image2] ... [Image20]]
            ...
            class 49 -> [[Image1],[Image2] ... [Image20]

    @return: 
        + a training set and 3 testing sets with pairs of images for the positive and negative class
        + respective dummy labels of the data of each set 
        + input_shape: the shape/dimensions of images
    '''  
    pairs = []
    labels = []
    length = len(num_classes)
    # Firsly, we get the minimum number of images 'min_num' from digit_indices to make sure that all created pairs having the same quantity 
    # min_num is for the loop because we will have adjecent pairs [i] & [i+1] 
    min_num = min([len(digit_indices[i]) for i in range(length)]) - 1
    for d in range(length):
        # Each loop, we create a postive pair and a negative pair with label [1,0]
        for i in range(min_num):
            # Create a positive pair
            indexP1, indexP2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[images[indexP1], images[indexP2]]]
            
            # Gets a random number between 1 and the length of the digits array then finds the modulus of d + the new random number
            # divided by the length of the digits array and assigns it to the variable dn
            rand = random.randrange(1, length)
            d_rand = (d + rand) % length
         
            # Create negative pairs
            indexN1, indexN2 = digit_indices[d][i], digit_indices[d_rand ][i]
            pairs += [[images[indexN1], images[indexN2]]]
            
            # Create label pairs
            labels += [0, 1]
    return np.array(pairs), np.array(labels,dtype='float32')


def get_pairs():
    '''
    This function to to get pairs of images from different training and testing sets, 
    using the above create_list_of_pairs to generate pairs for each set

    @return:
        + pairs of images used for training
        + pairs of images used for testing set 1
        + pairs of imagesused for testing set 2
        + pairs of images used for testing set 3
        + input_shape: the shape/dimension of images
    '''
    # get modified data
    train_images_1, train_labels_1, test_images_1, test_labels_1, test_images_2, test_labels_2, test_images_3, test_labels_3, class_names, full_images, full_labels, images_set_1, labels_set_1, input_shape = generate_data()

    #training set
    digit_indices = [np.where(train_labels_1 == num_classes_1[i])[0] for i in range(len(num_classes_1))]
    train_pairs_1, train_pairs_labels_1 = create_list_of_pairs(train_images_1, train_labels_1, digit_indices, num_classes_1)
    #testing set 1
    digit_indices = [np.where(test_labels_1 == num_classes_2[i])[0] for i in range(len(num_classes_2))]
    test_pairs_1, test_pairs_labels_1 = create_list_of_pairs(test_images_1, test_labels_1, digit_indices, num_classes_2)
    #testing set 2
    digit_indices = [np.where(test_labels_2 == num_classes_1[i])[0] for i in range(len(num_classes_1))]
    test_pairs_2, test_pairs_labels_2 = create_list_of_pairs(test_images_2, test_labels_2, digit_indices, num_classes_1)
    # testing set 3
    digit_indices = [np.where(test_labels_3 == i)[0] for i in range(len(class_names))]
    test_pairs_3, test_pairs_labels_3 = create_list_of_pairs(test_images_3, test_labels_3, digit_indices, class_names)

    global initiated
    # The following codes will be run only one (It will be skipped after the first execution.)
    if (initiated == False):
        print("---DATA VERIFICATION BEGINS--- ")
        print("*** Total images === ", full_images.shape[0])
        print("*** Total image labels === ", np.unique(full_labels))
        print("*** Images in original training set === ", images_set_1.shape[0])
        print("*** Image labels of the original training set === ", np.unique(labels_set_1))
        print("*** Images in original testing set=== ", test_images_1.shape[0])
        print("*** Image labels of original testing set === ", np.unique(test_labels_1))
        print("*** 80% of images in original training set will be used for training === ", train_images_1.shape[0])
        print("*** 80% of image labels of the original training set will be used for training  === ", np.unique(train_labels_1))
        print("*** 100% of images in original testing set will be used for testing set 1 === ", test_images_1.shape[0])
        print("*** 100% of image labels of the original testing set will be used for testing set 1 === ", np.unique(test_labels_1))
        print("*** 20% of images in original training set will be used for testing set 2 === ", test_images_2.shape[0])
        print("*** 20% of image labels of the original training set will be used for testing set 2 === ", np.unique(test_labels_2))
        print("*** 20% of images in the original training set + 100% of images in the original testing set will be used for testing set 3=== ", test_images_3.shape[0])
        print("*** 20% of image labels of the original training set + 100% of image labels of the original testing set will be used for testing set  === ", np.unique(test_labels_3))
    
        show_pair_images(train_pairs_1, "Training set")
        show_pair_images(test_pairs_1,  "Testing set 1")
        show_pair_images(test_pairs_2,  "Testing set 2")
        show_pair_images(test_pairs_3,  "Testing set 3")
        print("---DATA VERIFICATION ENDS---")
        print("****************************")
        initiated = True
    
    return (train_pairs_1, train_pairs_labels_1), (test_pairs_1, test_pairs_labels_1), (test_pairs_2, test_pairs_labels_2), (test_pairs_3, test_pairs_labels_3), input_shape


def show_pair_images(images, name):
    '''
    The function is to show pairs of images to ensure that the dateset is seperated correctly.  
    @param: images, name    
    '''
    print(name + " : " + "Pair of images in a same alphabet")
    plt.figure(figsize=(10,10))
    for i in range(2):
        plt.subplot(1,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[0][i].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    
    print(name + " : " + "Pair of images in different alphabets")
    plt.figure(figsize=(10,10))
    for i in range(2):
        plt.subplot(1,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[1][i].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    pass


def contrastive_loss(y_true, y_pred):
    '''
    This function is to calculate contrastive loss which is specified in the requirement. 
    @param:
      y_true : 0 if same equivalence class, 1 if different equivalence class. 
      y_pred : the distance value calculated by the 'euclidean_distance' function 
    @return
      the value of contrastive loss value
    '''                                          
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin_contrastive_loss - y_pred, 0))
    return K.mean(((1-y_true) * square_pred + y_true * margin_square)/2)


def test_contrastive_loss(y_true,y_pred):
    '''This function is to test if the contrasive loss calculated using Keras
    returns the same result as calculated uing mathematics formula
    @param:
        y_true: the target label -  equals 0 if the two images are similar and 1 otherwise
        y_predict: the Euclidean distance between the two image features 

    @return:
        SUCCESSFUL if the those 2 loss results are similar
        FAIL if the those 2 loss result are different 
    '''
    #calculate the loss uing numpy
    print("--------TEST CONTRASIVE LOSS-------")
    loss_np = ((1-y_true)*(y_pred**2)+y_true*(max(0,margin_contrastive_loss-y_pred)**2))/2
    loss_np_result = format(loss_np,'.4f')
    print('RESULT FOR CONTRASTIVE LOSS USING MATH FORMULA: ',loss_np_result)

    #calculate the loss using our function
    loss_tf = contrastive_loss(y_true, y_pred)
    with tf.compat.v1.Session() as sess:
        loss_tf = tf.constant(loss_tf)
        loss_tf_val = sess.run(loss_tf)
        loss_tf_result = format(loss_tf_val,'.4f')
        print('RESULT FOR CONTRASTIVE LOSS USING OUR FUNCTION: ',loss_tf_result)
        if(loss_np_result == loss_tf_result):
            print("CONTRASTIVE LOSS FUNCTION TESTED SUCCESSFULLY")
        else:
            print("CONTRASTIVE LOSS FUNCTION FAILED")



#****************************************************************************************************************************
#***************************************GENERATE TRIPLET IMAGES AND TRIPLET LOSS*********************************************
#****************************************************************************************************************************

def create_list_of_triplets(images, labels, digit_indices, num_classes):
    '''
    This fuction is to return a list of image triplets including the anchor image, image in a same class (positive), image in a different class (negative)
    @param:
        + images: the input images
        + labels: the class of the images (the alphabet that an image belongs to)
        + digit_indices: include the indexes of images in each class classified, for example:
            class 0 -> [[Image1],[Image2] ... [Image20]]
            class 1 -> [[Image1],[Image2] ... [Image20]]
            ...
            class 49 -> [[Image1],[Image2] ... [Image20]
        + num_classes: the number of classes

    @return:
        + an array of triplets and respective dummy labels 
    '''
    triplets = []
    labels = []
    length = len(num_classes)
    # Firsly, we get the minimum number of images 'min_num' from digit_indices to make sure that all created pairs having the same quantity 
    # min_num is for the loop because we will have adjecent pairs [i] & [i+1]
    min_num = min([len(digit_indices[i]) for i in range(length)]) - 1
    for d in range(length):
        # Each loop, we create a postive pair and a negative pair with label [1,0]
        for i in range(min_num):
            # Create anchor
            rand = random.randrange(1, length)
            d_rand = (d + rand) % length
            index_anchor = digit_indices[d][i]
            index_positive = digit_indices[d][i+1]
            index_negative= digit_indices[d_rand][i]
            triplets += [[images[index_anchor], images[index_positive],images[index_negative]]]
    #convert into dummy labels 
    labels = np.ones(len(triplets[:][:]))
    return np.array(triplets), np.array(labels,dtype='float32')


def get_triplets():
    '''
    This function to to get triplets from different training and testing sets, 
    using the above create_list_of_triplets to generate triplets for each set

    @return:
        + triplets used for training
        + triplets used for testing set 1
        + triplets used for testing set 2
        + triplets used for testing set 3
        + input_shape: the shape/dimensions of images
    '''
    # get modified data
    train_images_1, train_labels_1, test_images_1, test_labels_1, test_images_2, test_labels_2, test_images_3, test_labels_3, class_names, full_images, full_labels, images_set_1, labels_set_1, input_shape = generate_data()
    #training set
    digit_indices = [np.where(train_labels_1 == num_classes_1[i])[0] for i in range(len(num_classes_1))]
    train_triplets_1 = create_list_of_triplets(train_images_1, train_labels_1, digit_indices, num_classes_1)
    #testing set 1
    digit_indices = [np.where(test_labels_1 == num_classes_2[i])[0] for i in range(len(num_classes_2))]
    test_triplets_1 = create_list_of_triplets(test_images_1, test_labels_1, digit_indices, num_classes_2)
    #testing set 2
    digit_indices = [np.where(test_labels_2 == num_classes_1[i])[0] for i in range(len(num_classes_1))]
    test_triplets_2 = create_list_of_triplets(test_images_2, test_labels_2, digit_indices, num_classes_1)
    # testing set 3
    digit_indices = [np.where(test_labels_3 == i)[0] for i in range(len(class_names))]
    test_triplets_3 = create_list_of_triplets(test_images_3, test_labels_3, digit_indices, class_names)

    global initiated
    # The following codes will be run only one (It will be skipped after the first execution.)
    if (initiated == False):
        print("---DATA VERIFICATION BEGINS--- ")
        print("*** Total images === ", full_images.shape[0])
        print("*** Total image labels === ", np.unique(full_labels))
        print("*** Images in original training set === ", images_set_1.shape[0])
        print("*** Image labels of the original training set === ", np.unique(labels_set_1))
        print("*** Images in original testing set=== ", test_images_1.shape[0])
        print("*** Image labels of original testing set === ", np.unique(test_labels_1))
        print("*** 80% of images in original training set will be used for training === ", train_images_1.shape[0])
        print("*** 80% of image labels of the original training set will be used for training  === ", np.unique(train_labels_1))
        print("*** 100% of images in original testing set will be used for testing set 1 === ", test_images_1.shape[0])
        print("*** 100% of image labels of the original testing set will be used for testing set 1 === ", np.unique(test_labels_1))
        print("*** 20% of images in original training set will be used for testing set 2 === ", test_images_2.shape[0])
        print("*** 20% of image labels of the original training set will be used for testing set 2 === ", np.unique(test_labels_2))
        print("*** 20% of images in the original training set + 100% of images in the original testing set will be used for testing set 3=== ", test_images_3.shape[0])
        print("*** 20% of image labels of the original training set + 100% of image labels of the original testing set will be used for testing set  === ", np.unique(test_labels_3))
    
        show_triplet_images(train_triplets_1, "Training set")
        show_triplet_images(test_triplets_1,  "Testing set 1")
        show_triplet_images(test_triplets_2,  "Testing set 2")
        show_triplet_images(test_triplets_3,  "Testing set 3")
        print("---DATA VERIFICATION ENDS---")
        print("****************************")
        initiated = True
    
    return train_triplets_1, test_triplets_1, test_triplets_2, test_triplets_3, input_shape

def identity_loss(y_true, y_pred):
    '''
    This function is to calculate identity loss based on y_true which is created based on input of the create_list_of_triplets function. 
    @param:
       y_true: labels of images returned by create_list_of_triplets function
       y_pred: the prediction generated by the model 

    @return
      the value of identity loss
    '''   
    return K.mean(y_pred)

def triplet_loss(x):
    '''
    This function is to calculate triplet loss.  
    @param:
        x: [anchor,positive,negative]

    @return
      the value of triplet loss value
    '''                              
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+margin_triplet_loss
    loss = K.maximum(basic_loss,0.0)
    return loss

def show_triplet_images(images, name):
    '''
    The function is to show triplets of images to ensure that the dateset is seperated correctly.  
    @param: images, name    
    '''
    print(name + " : " + "Anchor Images")
    plt.figure(figsize=(10,10))
    for i in range(3):
        plt.subplot(1,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[0][0][i].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    
    print(name + " : " + "Positive Images")
    plt.figure(figsize=(10,10))
    for i in range(3):
        plt.subplot(1,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[0][1][i].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()

    print(name + " : " + "Negative Images")
    plt.figure(figsize=(10,10))
    for i in range(3):
        plt.subplot(1,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[0][2][i].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    pass


def test_triplet_loss(num_data):
    """
    This function is to test whether our triplet_loss function using Tensorflow 
    returns the correct result compared to the calculation using numpy. 
    @param: num_data: number of data to put in the embedding layer

    @return:
        SUCCESSFUL if the those 2 loss results are similar
        FAIL if the those 2 loss result are different 
    """
    global num_classes
    feature_dimension=num_classes
    embeddings = [np.random.rand(num_data, feature_dimension).astype(np.float32),
                  np.random.rand(num_data, feature_dimension).astype(np.float32),
                  np.random.rand(num_data, feature_dimension).astype(np.float32)]
    labels = np.random.randint(0, 1, size=(num_data)).astype(np.float32)
      
    #Compute loss with numpy
    loss_np = 0.
    anchor = embeddings[0]
    positive = embeddings[1]
    negative = embeddings[2]
    for i in range(num_data):
        pos_dist = np.sum(np.square(anchor[i] - positive[i]))
        neg_dist = np.sum(np.square(anchor[i] - negative[i]))
        loss_np += max(pos_dist-neg_dist+margin_triplet_loss, 0.)
    loss_np /= num_data

    loss_np_result = format(loss_np,'.4f')
    print('--------TEST TRIPLET LOSS-------')
    print('RESULT FOR TRIPLET LOSS USING NUMPY:', loss_np_result)

    loss_tf = Lambda(triplet_loss)(embeddings)
    loss_tf = identity_loss(_,loss_tf)
    with tf.compat.v1.Session() as sess:
        loss_tf = tf.constant(loss_tf)
        loss_tf_val = sess.run(loss_tf)
        loss_tf_result = format(loss_tf_val, '.4f')
        print('RESULT FOR TRIPLET LOSS USING OUR FUNCTION:', loss_tf_result)
        if(loss_np_result == loss_tf_result):
            print("TRIPLET LOSS FUNCTION TESTED SUCCESSFULLY")
        else:
            print("TRIPLET LOSS FUNCTION FAILED")



#****************************************************************************************************************************
#******************************************EXPLORE THE BEST HYPERPARAMETERS FOR CNN MODEL***************************************
#****************************************************************************************************************************


def explore_better_CNNlayer_parameters():
    '''
    The function is to explore suitable parameters for the CNN network. 
    We find the best network hyperparameters using GridSearchCV. 

    @param: train_images, train_labels, test_images, test_labels

    @return: the suggested hyperparameters for CNN model
    '''

    (ds_train, ds_test), ds_info = tfds.load(name='omniglot', splits=['train', 'test'], with_info=True)
    
    #get the image and labels of training and test sets, and reshape images into 28x28 dimensions 
    y_train = ([glyph["alphabet"] for glyph in ds_train])
    x_train = ([glyph["image"] for glyph in ds_train])
    x_train,y_train=resize_image(x_train,y_train)

    y_test= ([glyph["alphabet"] for glyph in ds_test])
    x_test = ([glyph["image"] for glyph in ds_test])
    x_test,y_test=resize_image(x_test,y_test)

    #convert labels and images into numpy
    y_train= np.array(y_train)
    x_train = np.array((x_train)[:, :, :,0])
    y_test= np.array(y_test)
    x_test = np.array((x_test)[:, :, :,0] )
    
    img_rows, img_cols = x_train.shape[1:3]
    num_classes = len(np.unique(y_train))
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)    
    
    # convert to float32 and rescale between 0 and 1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    epochs = 3
    batch_size = 128

    # Define CNN Networks
    def generate_CNN_model(filters, kernel_size, pool_size, dense_layer_size):
        '''
        Building the architecture of the model based on LeNet-5.  
        @param: 
            dense_layer_sizes: List of layer sizes to be chosen
            filters: Number of convolutional filters in each convolutional layer
            kernel_size: Convolutional kernel size
            pool_size: Size of pooling area for Max pooling

        @return: CNN model
        '''
        seq = Sequential()
        seq.add(Convolution2D(filters, kernel_size = kernel_size,
        activation = 'relu',
        input_shape = input_shape))
        seq.add(MaxPooling2D(pool_size = pool_size,
                strides = (2, 2),    
                padding = 'same'))
        seq.add(Convolution2D(filters, kernel_size = kernel_size, activation = 'relu')) 
        seq.add(MaxPooling2D(pool_size = pool_size,
                strides = (2, 2),    
                padding = 'same'))    
        seq.add(Flatten())
        seq.add(Dense(dense_layer_size, activation='relu')) 
        seq.add(Dense(dense_layer_size, activation='relu'))    
        seq.add(Dense(num_classes, activation='sigmoid'))
        seq.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
        return seq
    
    # Generate a classifier using the defined CNN model
    input_classifier = KerasClassifier(generate_CNN_model, verbose=1) 
    
    # Define values for the below hyperparameters
    param_grid = {'dense_layer_size': [64, 84, 120], 'filters': [32, 64, 128], 'kernel_size': [2, 3], 'pool_size': [2,3]}
    
    # Create a GridSearchCV validation to find the better hyperparameters 
    validator = model_selection.GridSearchCV(input_classifier, param_grid = param_grid, cv = 3,  verbose=0)   
    
    # Train the model to fit the OMNIGLOT dataset
    validator.fit(x_train, y_train,  
                      batch_size = batch_size,
                      epochs=epochs,
                      verbose=1) 
    
    # Output the best value for hyperparameters of CNN layers 
    print('\nThe best hypermeters for the model are: ')
    print(validator.best_params_)

    pass


#****************************************************************************************************************************
#*********************************BUILD DIFFERENT CNN MODELS WITH CUSTOMIZED HYPERPARAMETERS*********************************
#****************************************************************************************************************************

def create_base_network(input_shape):
    '''
    The customized combination of layers for the first network,applying the best hyperparameters from explore_better_CNNlayer_parameters() function,
    without Dropout and Flatten layer.
    '''
    seq = Sequential()
    # Partially connected layer
    seq.add(Convolution2D(128, kernel_size = (3, 3),
        activation = 'relu',
        input_shape = input_shape))
    # seq.add(BatchNormalization())
    seq.add(MaxPooling2D(pool_size = (2, 2)))
    seq.add(Convolution2D(256, 
        kernel_size = (3, 3), 
        activation = 'relu')) 
    # seq.add(BatchNormalization())
    seq.add(MaxPooling2D(pool_size = (2, 2)))
    seq.add(GlobalAveragePooling2D())
    seq.add(Flatten())
    seq.add(Dense(120, 
        activation='relu', 
        kernel_regularizer=l2(0.01), 
        bias_regularizer=regularizers.l1(0.01))) 
    seq.add(Dropout(0.25))
    seq.add(Dense(84, activation='relu'))
    seq.add(Dense(50, activation='sigmoid'))
    seq.summary()
    return seq

def show_plot(history):
    '''
    This function is to plot the accurary & loss graph for the comparison.
    @param:  model fit result (history) 
    '''
    #plot accuracy 
    plt.plot(history.history['accuracy'], color = "Black")
    plt.plot(history.history['val_accuracy'], color = "Red")
    plt.title('Siamese Network - Accuracy')
    plt.ylabel('Percent')
    plt.xlabel('Epoch')
    plt.legend(['train_acc','val_acc'],loc='lower right')
    plt.show()
    
    #plot loss 
    plt.plot(history.history['loss'], color = "Black")
    plt.plot(history. history['val_loss'], color = "Red")
    plt.title('Siamese Network - Loss')
    plt.ylabel('Percent')
    plt.xlabel('Epoch')
    plt.legend(['train_loss','val_loss'],loc='upper right')
    plt.show()
    pass



#****************************************************************************************************************************
#*************************************************BUILD AND TRAIN SIAMESE NETWORK********************************************
#****************************************************************************************************************************

def create_siamese_network(input_shape, loss_type):
    '''
    This function is to create Siamese network using contrasive loss/ triplet loss 
    @params: 
        +input_shape: the shape/dimensions of images
        +loss_type: 0 for contrasive loss, 1 for triplet loss

    @return: respective Siamese model 
    '''
    
    # Network initialisation 
    base_network = Sequential()
    base_network = create_base_network(input_shape)
    
    #for constrasive loss:
    if(loss_type == 0):
      # Initiate the shape for two tensors
      input_image_1 = Input(shape=input_shape)
      input_image_2 = Input(shape=input_shape)
    
      # Use the same base_network to input two tensors with sharing weights of the network
      processed_image_1 = base_network(input_image_1)
      processed_image_2 = base_network(input_image_2)
      
      # Lambda Layer for calculating two tensors by using Euclidian Distance
      distance = Lambda(euclidean_distance)([processed_image_1, processed_image_2])
      
      # Model Initialisation for contrasive loss 
      model = Model([input_image_1, input_image_2], distance)
      model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])

      #for triplet loss:
    else:
      # Initiate the shape for 3 tensors of triplets 
      input_anchor = Input(shape=input_shape)
      input_positive = Input(shape=input_shape)
      input_negative = Input(shape=input_shape)
    
      # Use the same base_network to input 3 tensors with sharing weights of the network
      anchor = base_network(input_anchor)
      positive = base_network(input_positive)
      negative = base_network(input_negative)
      #calculate the triplet_loss
      triplet_loss_value = Lambda(triplet_loss)([anchor, positive, negative]) 
      
      # Model Initialisation for triplet loss 
      model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=triplet_loss_value)
      model.compile(loss=identity_loss, optimizer=rms, metrics=[accuracy])
    return model


def train_network(epochs, loss_type=0):
    '''
    train Siamese network
    @params: 
        epochs: it refers training times.
        loss_type: 0 = contrastive loss / !=0 = triplet loss
    @return: model (Siamese Network)
    '''
    batch_size = 128
    # Training Siamese Network
    if(loss_type == 0):
        print("--------TRAINING SIAMESE MODEL WITH CONTRASTIVE LOSS--------")
        # Running Contrastive Loss:

        #get pairs of images for each data set
        (train_pairs_1, train_pairs_labels_1), (test_pairs_1, test_pairs_labels_1), (test_pairs_2, test_pairs_labels_2), (test_pairs_3, test_pairs_labels_3), input_shape = get_pairs()
        print(f"train_pairs_1.shape:{train_pairs_1.shape}")

        #create a siamese network
        model = create_siamese_network(input_shape, loss_type)

        #train the model 
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit([train_pairs_1[:, 0], train_pairs_1[:, 1]], train_pairs_labels_1,
                  batch_size=batch_size,
                  epochs=epochs, 
                  verbose=1,
                  validation_data=([test_pairs_1[:, 0], test_pairs_1[:, 1]], test_pairs_labels_1),
                  callbacks=[early_stopping_callback])

        #plot the accuracy and loss results of the trained model
        show_plot(history)

        #test the model with 3 different testing sets
        score_1 = model.evaluate([test_pairs_1[:, 0], test_pairs_1[:, 1]], test_pairs_labels_1, verbose = 0)
        print('TESTING SET 1: Test Loss = %0.2f%%' % (100 * score_1[0]))
        print('TESTING SET 1: Test Accuracy = %0.2f%%' % (100 * score_1[1]))
        
        score_2 = model.evaluate([test_pairs_2[:, 0], test_pairs_2[:, 1]], test_pairs_labels_2, verbose = 0)
        print('TESTING SET 2: Test Loss = %0.2f%%' % (100 * score_2[0]))
        print('TESTING SET 2: Test Accuracy = %0.2f%%' % (100 * score_2[1]))

        score_3 = model.evaluate([test_pairs_3[:, 0], test_pairs_3[:, 1]], test_pairs_labels_3, verbose = 0)
        print('TESTING SET 3: Test Loss = %0.2f%%' % (100 * score_3[0]))
        print('TESTING SET 3: Test Accuracy = %0.2f%%' % (100 * score_3[1]))


        pred1 = model.predict([test_pairs_1[:, 0],test_pairs_1[:, 1]])
        pred2 = model.predict([test_pairs_2[:, 0],test_pairs_2[:, 1]])
        pred3 = model.predict([test_pairs_3[:, 0],test_pairs_3[:, 1]])
        print(f"Confusion Matrix with testing set 1: {confusion_matrix(test_pairs_labels_1,pred1>=0.5)}")
        print(f"Confusion Matrix with testing set 2: {confusion_matrix(test_pairs_labels_2,pred2>=0.5)}")
        print(f"Confusion Matrix with testing set 3: {confusion_matrix(test_pairs_labels_3,pred3>=0.5)}")

    else:
        print("--------TRAINING SIAMESE MODEL WITH TRIPLET LOSS--------")
        # Running Contrastive Loss:

        #get triplets for each data set
        train_triplets_1, test_triplets_1, test_triplets_2, test_triplets_3, input_shape = get_triplets()
        
        #create a siamese network
        model = create_siamese_network(input_shape, loss_type)

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
        #train the model 
        history = model.fit([train_triplets_1[0][:,0],train_triplets_1[0][:,1],train_triplets_1[0][:,2]],train_triplets_1[1],
                  batch_size=batch_size,
                  epochs=epochs, 
                  verbose=1,
                  validation_data=([test_triplets_1[0][:,0],test_triplets_1[0][:,1],test_triplets_1[0][:,2]],test_triplets_1[1]),
                  callbacks=[early_stopping_callback])
       
        #plot the accuracy and loss results of the trained model
        show_plot(history)
        
        #testi the model with 3 different testing sets
        score_1 = model.evaluate([test_triplets_1[0][:,0],test_triplets_1[0][:,1],test_triplets_1[0][:,2]],test_triplets_1[1], verbose = 0)
        print('TESTING SET 1: Test Loss = %0.2f%%' % (100 * score_1[0]))
        print('TESTING SET 1: Test Accuracy = %0.2f%%' % (100 * score_1[1]))

        score_2 = model.evaluate([test_triplets_2[0][:,0],test_triplets_2[0][:,1],test_triplets_2[0][:,2]],test_triplets_2[1], verbose = 0)
        print('TESTING SET 2: Test Loss = %0.2f%%' % (100 * score_2[0]))
        print('TESTING SET 2: Test Accuracy = %0.2f%%' % (100 * score_2[1]))
      
        score_3 = model.evaluate([test_triplets_3[0][:,0],test_triplets_3[0][:,1],test_triplets_3[0][:,2]],test_triplets_3[1], verbose = 0)
        print('TESTING SET 3: Test Loss = %0.2f%%' % (100 * score_3[0]))
        print('TESTING SET 3: Test Accuracy = %0.2f%%' % (100 * score_3[1]))

        pred1 = model.predict([test_triplets_1[0][:,0],test_triplets_1[0][:,1],test_triplets_1[0][:,2]])
        pred2 = model.predict([test_triplets_2[0][:,0],test_triplets_2[0][:,1],test_triplets_2[0][:,2]])
        pred3 = model.predict([test_triplets_3[0][:,0],test_triplets_3[0][:,1],test_triplets_3[0][:,2]])
        print(f"Confusion Matrix with testing set 1: {confusion_matrix(test_triplets_1[1],pred1>=0.5)}")
        print(f"Confusion Matrix with testing set 2: {confusion_matrix(test_triplets_2[1],pred2>=0.5)}")
        print(f"Confusion Matrix with testing set 3: {confusion_matrix(test_triplets_3[1],pred3>=0.5)}")

    pass


#------------------------\
# III. MAIN FUNCTION      |
#------------------------/

def main_func():

    ### Uncomment the below function to explore the better parameters of CNN layers:
    ### The result shows that the suggested best hyperparameters are: dense_layer_size=84, filter=64, kernel_size=3, pool_size=2
    # explore_better_CNNlayer_parameters()
    ### Uncomment the below two functions to test the results of our two loss functions:
    # test_contrastive_loss(1.0,0.1) 
    # test_triplet_loss(100)
    ### Uncomment the below functions to train Siamese model with different parameters:
    ### train_network(mnumber of epochs, 0 for contrastive loss/ 1 for triplet loss
    ## *TRAINING WITH CONTRASTIVE LOSS
    # train_network(10, 0)
    ## *TRAINING WITH TRIPLET LOSS
    train_network(10, 1)
    pass

if __name__ == '__main__':
    main_func()
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \
#                                   THE END                                    |                                                 
#       Students: Huynh Ngoc Tram Nguyen, Filipe Arena, Viet Thang Doan        |
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -/
