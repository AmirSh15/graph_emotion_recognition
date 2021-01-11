# This code load RML database

# ==================================
#              Imports
# ==================================
import os
import numpy as np
import cv2
import face_alignment
import shutil


# ==================================
#           Functions
# ==================================
def get_voxceleb2_datalist(data_path, path):
    with open(path) as f:
        strings = f.readlines()
        a = strings[0].split()
        videolist = np.array([os.path.join(data_path, string.split()[0]) for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return videolist, labellist


# ==================================
#       Get Train/Val.
# ==================================
print('Loading data...')

trnlist, trnlb = get_voxceleb2_datalist(data_path='/home/user/Downloads/RML_Dataset', path='./RML.txt')

partition = {'train': trnlist.flatten()}
labels = {'train': trnlb.flatten()}


# ==================================
#       Extracting Landmarks
# ==================================

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

num_class = 6
fix_length = 30 * 3 

# Landmark file names
Graph_train_data = np.zeros(shape=(1, fix_length, 68, 2))
train_label = np.zeros(shape=(1, 1))

q = 0
c = 0  # counter of all data
c_p = 0  # counter of persons
a = np.zeros(1)
for Name in partition['train']:

    cap = cv2.VideoCapture(str(Name))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    train_data = np.zeros(shape=(1, 68, 2))

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            dir = 'RML_data/' + str(q) + '/'
            if (not os.path.isdir(dir)):
                os.mkdir(dir)
            cv2.imwrite(dir + str(c_p) + '.jpg', frame)
            c_p += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    print('Number of files:'+str(c))



# Save Graph data
np.save('train_graph_data_RML.npy', Graph_train_data[1:])
np.save('train_graph_label_RML.npy', train_label[1:])
