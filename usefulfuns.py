from mtcnn.mtcnn import MTCNN
import cv2
import os


dataset = r"D:\user\sockets_web\person"

# detect face using the mtcnn algorithm


def extract_mtcnn_faces(image, threshold=0.7):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_images = []
    for face in faces:
        x, y, w, h = face['box']
        if face['confidence'] > threshold:
            face_image = image[y:y+h, x:x+w]
            if face_image.size != 0:
                face_images.append(face_image)
    return face_images


# detect face using the dnn
def extract_faces_dnn(image, DNN="CAFFE"):
    net = None
    if DNN == "CAFFE":
        modeFile = "models/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modeFile)
    elif DNN == "TF":
        modeFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modeFile, configFile)

    if net is None:
        print("Error loading network")
        exit()
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0, size=(
        300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        print(confidence)
        if confidence > 0.8:
            x1 = int(detections[0, 0, i, 3] * image.shape[1])
            y1 = int(detections[0, 0, i, 4] * image.shape[0])
            x2 = int(detections[0, 0, i, 5] * image.shape[1])
            y2 = int(detections[0, 0, i, 6] * image.shape[0])
            face = image[y1:y2, x1:x2]
            bboxes.append(face)
    return bboxes


# detect face using haar cascade
def extract_faces(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)
    faces_images = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        if face.size != 0:
            face = cv2.resize(face, (100, 100))
            faces_images.append(face)
    return faces_images


# Loop through the images in the full image dataset and extract faces and append them to same folder
for folder in os.listdir(dataset):
    folder_path = os.path.join(dataset, folder)
    image_paths = [os.path.join(folder_path, image_file)
                   for image_file in os.listdir(folder_path)]
    for image_file in image_paths:
        image = cv2.imread(image_file)
        faces = extract_faces_dnn(image)
        for idx, face in enumerate(faces):
            # save the face in the same folder
            face_filename = f"{os.path.basename(image_file)}_face_{idx}.jpg"
            face_path = os.path.join(folder_path, face_filename)
            cv2.imwrite(face_path, face)
