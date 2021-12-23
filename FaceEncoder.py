import numpy as n
import sklearn
import face_recognition
import cv2
import dlib
import pickle  # it used to pick up a bunch of image files from 1 path
from PIL import Image
import glob  # for picking up a bunch of image


class ImagesEncoder:
    def __init__(self, encode_name, target):

        image_list = []
        faces_encodings = []

        print("Importing images......")

        image_target = target + "/*.jpg"
        for filename in glob.glob(image_target):
            im = Image.open(filename)
            image_list.append(im)

        print("Images importing procedure complete. ", "There are ", len(image_list), " images!")

        # image = cv2.imread(image_list[0].filename) # try to display the path for testing

        # initialize the list of known encodings and known names

        for (i, imagePath) in enumerate(image_list):
            # print(imagePath.filename)
            # extract the person name from the image path
            print("processing image {}/{}/{}".format(i + 1, len(image_list),imagePath))
            # load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
            image = cv2.imread(imagePath.filename)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # dlib assumes rgb ordering rather than OpenCV’s default BGR.

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb, 1, model="hog")
            print(boxes)
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # encoding returning list of ndArray which is 128-D

            # build a d(dictionary) of the image path, bounding box location, facial encodings
            # the code below simply pack path+location+encoding into ONE set and put every packet into a dictionary.
            # Question 2: why we use for loop here? because there might be more than 1 faces in an image :-)
            d = [{"imagePath": imagePath.filename, "loc": box, "encoding": enc}
                 for (box, enc) in zip(boxes, encodings)]
            # print(d[1])       # we can print the d out to see what is the above code going on
            faces_encodings.extend(d)  # save every dictionary to our facesEncodings list

        print("Serializing encodings...")
        encoding_file_name = encode_name + ".pickle"
        f = open(encoding_file_name, 'wb')
        f.write(pickle.dumps(faces_encodings))
        f.close()
        print("Serializing Encodings File saved! ")
