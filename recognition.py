from PIL import Image
import numpy
import json
import cv2
import csv
import os


class FaceRecognition:
    def __init__(self,
                 dataset_path='dataset/',
                 groups_path='groups/',
                 visits_folder='visits/',
                 trainer_path='trainer/',
                 face_haar_path='haarcascade_frontalface_default.xml',
                 width=640,
                 height=480, ):
        self._dataset_path = dataset_path
        self._groups_path = groups_path
        self._visits_folder = visits_folder
        self._trainer_path = trainer_path
        self._face_haar_path = face_haar_path
        self._width = width
        self._height = height

    def add_group(self, persons, group_name):
        self.set_dataset(persons)
        print('Successfully set the dataset!')
        self.train(group_name, persons)

    def set_dataset(self, persons, purge=True):
        cam = cv2.VideoCapture(0)
        cam.set(3, self._width)  # set video width
        cam.set(4, self._height)  # set video height

        if purge:
            for f in os.listdir(self._dataset_path):
                os.remove(os.path.join(self._dataset_path, f))

        face_detector = cv2.CascadeClassifier(self._face_haar_path)

        for person in persons:
            person_id = list(person.keys())[0]
            person_name = person[person_id]
            count = 0

            while True:
                pre_img = numpy.full((self._height, self._width, 3), 0, dtype=numpy.uint8)
                cv2.putText(pre_img, 'Please, press enter when ', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(pre_img, person_name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(pre_img, 'will look to the camera.', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

                cv2.imshow('Vision', pre_img)
                k = cv2.waitKey()
                if k == 13:
                    break

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                if len(faces) == 0:
                    count = 0
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1

                    # Save the captured image into the datasets folder
                    cv2.imwrite(self._dataset_path + "Person." + str(
                            person_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                    cv2.putText(img, person_name + ", we'll take some photos...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                    cv2.imshow('Vision', img)

                k = cv2.waitKey(100) & 0xff
                if k == 27:
                    break
                elif count >= 30:
                    break

        cam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_images_and_labels(detector, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for imagePath in image_paths:
            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = numpy.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return face_samples, ids

    def train(self, group_name, persons):
        self.save_dict_json(group_name, persons)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(self._face_haar_path)

        print("Training... Please wait a few seconds...")
        faces, ids = self.get_images_and_labels(detector, self._dataset_path)
        recognizer.train(faces, numpy.array(ids))

        # Save the model
        recognizer.write(self._trainer_path + group_name + '.yml')

        print("{0} faces trained".format(len(numpy.unique(ids))))

    def recognize(self, group_name):
        recognized = set()

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self._trainer_path + group_name + '.yml')
        face_cascade = cv2.CascadeClassifier(self._face_haar_path);

        font = cv2.FONT_HERSHEY_SIMPLEX
        persons = self.get_dict_json(group_name)

        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:
            ret, img = cam.read()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                name = 'unknown'
                # Check if confidence is less them 100 ==> "0" is perfect match
                if confidence < 100:
                    recognized.add(id)
                    name = persons[str(id)]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.putText(img, 'Press ESC to end and save', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.imshow('Vision', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        return recognized, persons

    def save_dict_json(self, group_name, persons):
        persons = {k: v for d in persons for k, v in d.items()}
        with open(self._groups_path + group_name + '.json', 'w') as fp:
            json.dump(persons, fp, indent=4)
        return True

    def get_dict_json(self, group_name):
        with open(self._groups_path + group_name + '.json') as f:
            data = json.load(f)

        return data

    def save_list_csv(self, lists, name, folder=''):
        folder = folder if len(folder) > 0 else self._visits_folder

        with open(folder + name + '.csv', mode='w') as file:
            file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            [file.writerow(row) for row in lists]

        return True
