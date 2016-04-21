import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection 

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector

"""
pop up an image showing the mouth with a blue rectangle
"""
def show(area): 
    cv.Rectangle(img,(area[0][0],area[0][1]),
                     (area[0][0]+area[0][2],area[0][1]+area[0][3]),
                    (255,0,0),2)
    #cv.NamedWindow('Face Detection', cv.CV_WINDOW_NORMAL)
    #cv.ShowImage('Face Detection', img) 
    #cv.WaitKey()

"""
given an area to be cropped, crop() returns a cropped image
"""
def crop(area): 
    crop = img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] #img[y: y + h, x: x + w]
    return crop

"""
given a jpg image, vectorize the grayscale pixels to 
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""
def vectorize(filename):
    size = WIDTH, HEIGHT # (width, height)
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array


if __name__ == '__main__':
    """
    load training data
    """
    # create a list for filenames of smiles pictures
    smilefiles = []
    with open('smiles.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            smilefiles += rec

    # create a list for filenames of neutral pictures
    neutralfiles = []
    with open('neutral.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load smile data
    PATH = "../data/smile/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(1)

    # load neutral data    
    PATH = "../data/neutral/"
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize(PATH + filename)
        labels.append(0)
    #load sad data
    '''PATH = "../data/sad/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(2)'''

    """
    training the data with logistic regression
    """
    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    

    """
    open webcam and capture images
    """
   
    img = cv2.imread("/home/qbuser/Desktop/4.jpeg") # input image
    mouth = mouthdetection.findmouth(img)
            # show(mouth)
    #if mouth != 2: # did not return error
                #mouthimg = crop(mouth)
    cv2.imwrite("webcam-m.jpg", mouth)
                # predict the captured emotion
    result = lr.predict(vectorize('webcam-m.jpg'))
    if result == 1:
                    print "you are smilinggggggggg! :-) "
    elif result == 0:
                    print "you are neutral "
    #elif result==2:
		    print "you are sad"
    #else:
     #           print "failed to detect mouth. Try hold your head straight and make sure there is only one face."
    
    cv2.destroyWindow("preview")
