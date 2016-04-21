import cv
import cv2
"""
input: a loaded image; 
output: [[x,y],[width,height]] of the detected mouth area
"""
def findmouth(img):
 mouth_cascade = cv2.CascadeClassifier('/home/qbuser/Desktop/haarcascade_mcs_mouth.xml')

 if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')


 #img=cv2.imread(img)
 mouth_rects = mouth_cascade.detectMultiScale(img,1.7, 11)
#print mouth_rects
 for (x,y,w,h) in mouth_rects:
        #y = int(y - 0.15*h)					
        #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
	cropped = img[y:y+h ,x:x+w]
	#cv2.imshow("cropped", cropped)
	#cv2.waitKey(0)
	cv2.imwrite('/home/qbuser/Desktop/cr.jpg',cropped)

 return cropped

'''def findmouth(img):

  # INITIALIZE: loading the classifiers
  haarFace = cv.Load('haarcascade_frontalface_default.xml')
  haarMouth = cv.Load('haarcascade_mouth.xml')
  # running the classifiers
  storage = cv.CreateMemStorage()
  detectedFace = cv.HaarDetectObjects(img, haarFace, storage)
  detectedMouth = cv.HaarDetectObjects(img, haarMouth, storage)

  # FACE: find the largest detected face as detected face
  maxFaceSize = 0
  maxFace = 0
  if detectedFace:
   for face in detectedFace: # face: [0][0]: x; [0][1]: y; [0][2]: width; [0][3]: height 
    if face[0][3]* face[0][2] > maxFaceSize:
      maxFaceSize = face[0][3]* face[0][2]
      maxFace = face
  
  if maxFace == 0: # did not detect face
    return 2

  def mouth_in_lower_face(mouth,face):
    # if the mouth is in the lower 2/5 of the face 
    # and the lower edge of mouth is above that of the face
    # and the horizontal center of the mouth is the center of the face
    if (mouth[0][1] > face[0][1] + face[0][3] * 3 / float(5) 
      and mouth[0][1] + mouth[0][3] < face[0][1] + face[0][3]
      and abs((mouth[0][0] + mouth[0][2] / float(2)) 
        - (face[0][0] + face[0][2] / float(2))) < face[0][2] / float(10)):
      return True
    else:
      return False

  # FILTER MOUTH
  filteredMouth = []
  if detectedMouth:
   for mouth in detectedMouth:
    if mouth_in_lower_face(mouth,maxFace):
      filteredMouth.append(mouth) 
  
  maxMouthSize = 0
  for mouth in filteredMouth:
    if mouth[0][3]* mouth[0][2] > maxMouthSize:
      maxMouthSize = mouth[0][3]* mouth[0][2]
      maxMouth = mouth
      
  try:
    return maxMouth
  except UnboundLocalError:
    return 2
'''
