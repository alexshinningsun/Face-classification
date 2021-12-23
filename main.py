import numpy as n
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import cv2
import dlib
from imutils import build_montages
import pickle   # it used to pick up a bunch of image files from 1 path
from PIL import Image
import os
import FaceEncoder

"""
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor.dat")
facerec = dlib.face_recognition_model_v1("dlib_resnet_model_v1.dat")

descriptors = []
images = []
pickle_encodings = []

# Now find all the faces and compute 128D face descriptors for each face.
for f in glob.glob(os.path.join("dataset/", "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    boxes = []
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.right(), d.bottom(), d.top()))
        # tp = (d.left(), d.right(), d.bottom(), d.top())
        # boxes.append(tp)

    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        tp = (d.left(), d.right(), d.bottom(), d.top())
        boxes.append(tp)
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape))

        pik = [{"imagePath": f, "loc": box, "encoding": enc}
             for(box, enc) in zip(boxes, face_descriptor)]
        pickle_encodings.extend(pik)

print("Serializing encodings...")
f = open('encodings.pickle', 'wb')
f.write(pickle.dumps(pickle_encodings))
f.close()
print("Serializing Encodings File saved! ")

"""
"""
larger_dataset_Images = FaceEncoder.ImagesEncoder(encode_name="encoding",target="dataset")

smaller_dataset_Images = FaceEncoder.ImagesEncoder(encode_name="subset_encoding",target="sub_dataset")
"""
try:
    os.mkdir('result')
except FileExistsError:
    print()
print("Importing encodings......")

subset_pickleFile = pickle.loads(open("subset_encoding.pickle", "rb").read()) # load the encoding file
subset_pickleFile = n.array(subset_pickleFile)    # turn encoding file into numpy for easily access like (for loop)
sub_encodings = [d["encoding"] for d in subset_pickleFile] # fetch 128-D encodings

pickleFile = pickle.loads(open("encoding.pickle", "rb").read()) # load the encoding file
pickleFile = n.array(pickleFile)    # turn encoding file into numpy for easily access like (for loop)
encodings = [d["encoding"] for d in pickleFile] # fetch 128-D encodings

# clustering
print("Clustering......")
dbClus = DBSCAN(eps=0.32, min_samples=2, n_jobs=6)        # eps=0.32 or 31 for the task, mine 0.39
dbClus.fit(encodings)                # # encodings

sub_dbClus = DBSCAN(eps=0.43, min_samples=2, n_jobs=6)        # eps=0.32 or 31 for the task, mine 0.39
sub_dbClus.fit(sub_encodings)                # # encodings

# determine the total number of unique faces found in the dataset
# find the unique faces/unique label IDs by NumPy’s unique  function
labelIDs = n.unique(dbClus.labels_)     # contains the label ID for all faces in our dataset
numUniqueFaces = len(n.where(labelIDs > -1)[0])
sub_labelIDs = n.unique(sub_dbClus.labels_)
sub_numUniqueFaces = len(n.where(sub_labelIDs > -1)[0])
# numUniqueFaces -1 == outliers ( too far away from any other clusters)
print("# of unique faces: {}".format(numUniqueFaces))

main_lbl = []       # label for main class
sub_lbl = []
main_file_path = []
sub_file_path = []

# loop over the unique face integers
for labelID in labelIDs:            # #  labelIDs sub_labelIDs
    # find all indexes into the `data` array that belong to the
    # current label ID, then randomly sample a maximum of 25 indexes
    # from the set
    # print(" faces for face ID: {}".format(labelID))
    idxs = n.where(dbClus.labels_ == labelID)[0]        # #  dbClus sub_dbClus
    # print("n.where: ", idxs)
    # grab a random sample of at most 25 images to include in the montage.
    # # idxs = n.random.choice(idxs, size=min(100, len(idxs)), replace=False)
    # initialize the list of faces to include in the montage
    faces = []

    # print(" n.random.choice: ", idxs)
    # print(len(idxs))        # print out number of faces in each ID
    new_file_name = ""
    print("labelID: ", labelID)
    new_file_name = str(labelID)
    try:
        os.mkdir("result/"+new_file_name)
    except FileExistsError:
        print("folder existed: ", new_file_name)
    for i in idxs:
        # print(pickleFile[i]["imagePath"].split(chr(92))[-1])       # # pickleFile subset_pickleFile
        main_file_path.append((pickleFile[i]["imagePath"]).split(chr(92))[-1])  # # pickleFile subset_pickleFile
        main_lbl.append(labelID)
        # load the input image and extract the face ROI using the bounding box coordinates
        image = cv2.imread(pickleFile[i]["imagePath"]) # #  pickleFile subset_pickleFile
        (top, right, bottom, left) = pickleFile[i]["loc"] # # pickleFile subset_pickleFile
        face = image[top:bottom, left:right]
        # force resize the face ROI to 96x96 and then add it to the
        # faces montage list
        face = cv2.resize(face, (96, 96))       # 96, 96
        faces.append(face)  # For visualize each cluster.

        # im_clu = Image.open((pickleFile[i]["imagePath"]), mode="r")
        # im_clu.save("result/"+new_file_name + '/'+(pickleFile[i]["imagePath"]).split(chr(92))[-1])
        # im_clu.close()
        cv2.imwrite("result/" + new_file_name + '/'+(pickleFile[i]["imagePath"]).split(chr(92))[-1]
                    , face, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    # aka: generate a single image montage  containing a 5×5 grid of faces
    montage = build_montages(faces, (96, 96), (10, 10))[0]  # from library imutils
    # show the output montage
    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    # cv2.imshow(title, montage)
    # cv2.waitKey(0)

for labelID in sub_labelIDs:  # sub_labelIDs
    # find all indexes into the `data` array that belong to the
    # current label ID, then randomly sample a maximum of 25 indexes
    # from the set
    print(" faces for face ID: {}".format(labelID))
    idxs = n.where(sub_dbClus.labels_ == labelID)[0]
    print("n.where: ", idxs)
    # grab a random sample of at most 25 images to include in the montage.
    # # idxs = n.random.choice(idxs, size=min(100, len(idxs)), replace=False)
    # initialize the list of faces to include in the montage
    faces = []

    print(" n.random.choice: ", idxs)
    print(len(idxs))  # print out number of faces in each ID

    for i in idxs:
        # print(subset_pickleFile[i]["imagePath"].split(chr(92))[-1])
        sub_file_path.append((subset_pickleFile[i]["imagePath"]).split(chr(92))[-1])
        sub_lbl.append(labelID)
        # load the input image and extract the face ROI using the bounding box coordinates
        image = cv2.imread(subset_pickleFile[i]["imagePath"])
        (top, right, bottom, left) = subset_pickleFile[i]["loc"]
        face = image[top:bottom, left:right]
        # force resize the face ROI to 96x96 and then add it to the
        # faces montage list
        face = cv2.resize(face, (96, 96))  # 96, 96
        faces.append(face)  # For visualize each cluster.

temp = n.zeros(len(main_file_path))
temp = temp -1
print("the length of dbClus.labels_: ", len(dbClus.labels_))
print("the length of sub_dbClus.labels_: ", len(sub_dbClus.labels_))
print("the length of main_lbl: ", len(main_lbl))
print("the length of sub_lbl: ", len(sub_lbl))

for i in range(len(main_file_path)):
    stt = ""
    stt = stt + main_file_path[i]
    split_str = stt.split(chr(92))[-1]
    if main_file_path[i] in sub_file_path:
        temp[i] = main_lbl[i]

print(main_lbl, len(main_lbl))
print(temp, len(temp))
print(classification_report(temp, main_lbl)) # pos_label = 1, labels=[1], average ='macro'
