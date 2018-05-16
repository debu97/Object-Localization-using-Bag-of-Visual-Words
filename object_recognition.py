# -*- coding: utf-8 -*-
import matplotlib as matplotlib
import glob2
import imageio
import numpy as np
from matplotlib import pylab as plt
import matplotlib.cm as cm
import math
import scipy
#from numpy import linalg as LA

def norm(bin):
    total = 0
    norm = [0] * 9
    norm = np.array(norm, dtype=float)
    for r in range(len(bin)):
        total += bin[r]
    # print(total)
    for r in range(len(bin)):
        norm[r] = bin[r] / total
    return norm


def Euclidean_distance(feat_one, feat_two):
    squared_distance = 0
    squared_distance = np.array(squared_distance,dtype = float)
  #   Assuming correct input to the function where the lengths of two features are   the same
    for i in range(len(feat_one)):
        squared_distance += (np.subtract(feat_one[i],feat_two[i]))**2
    ed = np.sqrt(squared_distance)
    return ed;



# Loading dataset

# -----------------Feature extraction------------------------
# ------------------Corner detection-------------------------
#
no_of_corners = 15
import cv2
cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('facecorner.avi',fourcc, 20.0, (640,480) )
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    #  cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
    corners = cv2.goodFeaturesToTrack(gray, no_of_corners, 0.05, 25)
    corners = np.int32(corners)
    img = frame
    for item in corners:
        x, y = item[0]
        cv2.circle(img, (x, y), 5, 255, -1)

    # out.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Top 'k' features", img)
    cv2.waitKey()
    cv2.imshow('frame', gray)

# When everything done, release the capture
cap.release()
# out.release()
cv2.destroyAllWindows()
# print(corners)

feat_space = []
feat_space = np.array(feat_space, dtype = float)

## --------HOG interest point descriptor--------

for corner in corners:
    w = 40
    Y_len, X_len = np.size(img, 0),np.size(img, 1)
    y1,y2,x1,x2 = corner[0][0] - w,corner[0][0] + w ,corner[0][1] - w,corner[0][1] + w
    if x1<0:
        x1 = 0
    elif x2>X_len:
        x2 = X_len
    elif y1>Y_len:
        y1 = Y_len
    elif y2<0:
        y2 = 0
    # print(corner[0][0],corner[0][1])
    # print(x1,x2,y1,y2)
    image_patch = img[x1:x2,y1:y2]
    #cv2.imshow('patches', image_patch)
    #cv2.waitKey()
    gx = cv2.Sobel(image_patch, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image_patch, cv2.CV_32F, 0, 1, ksize=1)
    # print(gx,gy)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # mag, angle = np.array(mag, dtype=int), np.array(angle, dtype=int)
    blue_bin = [0]*9
    blue_bin = np.array(blue_bin, dtype = float)
    green_bin = [0] * 9
    green_bin = np.array(green_bin, dtype = float)
    red_bin = [0] * 9
    red_bin = np.array(red_bin, dtype = float)
    # print(mag)
    # print("next")
    # print(angle)

    # -------------- BGR CHANNEL------------
    for l in range(3):
        for i in range(len(angle)):
            for j in range(len(angle[0])):
                for n in range(9):
                    if angle[i][j][l] >= 20*n and angle[i][j][l] < 20*(n+1):
                        if l==0:
                            blue_bin[n] += mag[i][j][l]
                        elif l==1:
                            green_bin[n] += mag[i][j][l]
                        elif l==2:
                            red_bin[n] += mag[i][j][l]

    # print(blue_bin)
    # print(green_bin)
    # print(red_bin)

    feat_vect = []
    feat_vect = np.array(feat_vect, dtype = float)
    blue_norm = [0] * 9
    green_norm = [0] * 9
    red_norm = [0] * 9

    blue_norm = np.array(blue_norm, dtype=float)
    green_norm = np.array(green_norm, dtype=float)
    red_norm = np.array(red_norm, dtype=float)

    blue_norm = norm(blue_bin)
    green_norm = norm(green_bin)
    red_norm = norm(red_bin)
    #print(blue_norm)
    #print(green_norm)
    #print(red_norm)

    bgr_norm = [0] * 9
    bgr_norm = np.array(bgr_norm, dtype = float)
    for n in range(9):
        bgr_norm[n] = np.sqrt(((blue_norm[n])** 2) + ((green_norm[n])** 2) + ((red_norm[n])** 2))
        blue_norm[n] = blue_norm[n]/bgr_norm[n]
        green_norm[n] = green_norm[n] / bgr_norm[n]
        red_norm[n] = red_norm[n] / bgr_norm[n]
        feat_vect = np.append(feat_vect,[blue_norm[n],green_norm[n],red_norm[n]])            
    #print(feat_vect)
        
    feat_space = np.append(feat_space,feat_vect)  
    
    # cv2.imshow('Horizontal gradient', gx)
    # cv2.waitKey()
    # cv2.imshow('magnitude', mag)
    # cv2.waitKey()

    # plt.hist(bin)
    # plt.xlabel('angle')
    # plt.ylabel('magnitude')
    # plt.show()

# cv2.imshow('patches', image_patch)
#print(image_patch.shape)
feat_space = np.reshape(feat_space,(len(corners), 27))
#print('Feature Space')
print(feat_space)

# ---------------------------------------------------K means--------------------------------------------------------------

dis = [0]*(len(corners)-1)
dis = np.array(dis, dtype = object)
for i in range(len(corners)):
    if(i!=(len(corners)-1)):
        dis[i] = Euclidean_distance(feat_space[i],feat_space[i+1])
#print(dis)

k = 3
centroids = [[0]*27]*k
centroid_old = [[0]*27]*k
classes = [[0] for i in range(len(feat_space))]
centroids = np.array(centroids, dtype = float)
centroid_old = np.array(centroid_old, dtype = float)
classes = np.array(classes, dtype = int)
for i in range(k):
    for j in range(27):
         centroids[i][j] = feat_space[i][j]        

#centroid_old = centroids
print(centroids)
fig1 = plt.figure(figsize = (5,5))
#plt.scatter([feat_space[0].mean()], [feat_space[1].mean()], color = 'k')
plt.scatter([centroids[0].mean()],[centroids[1].mean()], color = 'r')
plt.scatter([centroids[1].mean()],[centroids[2].mean()], color = 'g')
plt.scatter([centroids[2].mean()],[centroids[0].mean()], color = 'b')    
plt.xlim(0.56,0.59)
plt.ylim(0.56,0.59)

plt.show()

df = [0]*k
diff = [0]*k
df = np.array(df, dtype = object)
diff = np.array(diff, dtype = object)
print('Centroids')
max_iterations = 6
for i in range(max_iterations): 
    #print(centroids)
    for m in range(len(feat_space)):
        for centroid in range(k):
            df[centroid] = Euclidean_distance(feat_space[m],centroids[centroid]) # finding distance of a data point from all centroids
            #print(df[centroid])
        #print('Now check for min')        
        minimum = df[0]
        for l in range(len(df)):
            minimum = np.minimum(minimum,df[l])
        #print(minimum)
        for l in range(len(df)):
            if (df[l] == minimum):
                classes[m] = l       # assigning a data point to the closest centroid cluster. '0' means centroid 0.
    #print(classes)
    #print(centroids)
    # Centroid update
    count = [0]*k
    #count = np.array(count,dtype = float)
    cent_tot = [[0]*27]*k
    cent_tot = np.array(cent_tot,dtype = object)
    
    centroid_new = [[0]*27]*k
    centroid_new = np.array(centroid_new,dtype = object)

    for cluster in range(len(classes)):
        for j in range(k):
            if classes[cluster][0] == j:
                count[j] += 1
                cent_tot[j] += feat_space[cluster]     

    print(count)
    for j in range(k):    
        if (count[j]!=0):
            centroid_new[j] = cent_tot[j]/count[j]
            centroids[j] = centroid_new[j]

    #print(centroids)

fig2 = plt.figure(figsize = (5,5))
plt.scatter([centroids[0].mean()],[centroids[1].mean()], color = 'r')
plt.scatter([centroids[1].mean()],[centroids[2].mean()], color = 'g')
plt.scatter([centroids[2].mean()],[centroids[0].mean()], color = 'b')  
plt.xlim(0.56,0.59)
plt.ylim(0.56,0.59)
plt.show()

#--------------------Labelling of classes---------------------------------------

#clustered_image = [[[0]*3]*80]*80
print(classes)
class_prob = [0]*k
class_prob = np.array(class_prob,dtype = float)

total = 0;
for item in count:
    total += item
#print(total)
count = np.array(count,dtype = float)

for i in range(k):
    class_prob[i] = count[i]/total

#print(class_prob)
maximum = class_prob[0]
for l in range(k):
    maximum = np.maximum(maximum,class_prob[l])
#print(maximum)
location = []
location = np.array(location,dtype = int)
for l in range(k):
    if (class_prob[l] == maximum):
        print("object "+str(l)+" detected with " + str(count[l]) + " corner points")
        for x in range(no_of_corners):
            if (classes[x][0] == l):
                location = np.append(location, x)

#---------------------Object Localization---------------------------------------------
cv2.imshow("Object Localization", img)
print(location)
print(corners)

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #  cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
    corners = cv2.goodFeaturesToTrack(gray, no_of_corners, 0.05, 25)
    corners = np.int32(corners)
    img = frame
    for item in location:
        x, y = corners[item][0]
        cv2.circle(img, (x, y), 5, 255, -1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Object Localization", img)
    cv2.waitKey()
    cv2.imshow('frame', gray)

cap.release()
cv2.destroyAllWindows()


