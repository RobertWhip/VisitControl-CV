# VisitControl-CV
Create groups of people and control them visiting your classes. 

Results are stored in .csv files in visits/ folder.

Requirements:
1. numpy
2. cv2
3. Python3.x
4. Camera
5. other default python modules(PIL from Image, csv, os...)

### How to create group of people
1. Run python script main.py.
2. Enter "y" when the scripts asks "Add a new group to the Visit Control system? y/N:".
3. Enter the name of the group you want to add when the script asks "New group: ". For example: MyGroup1.
4. Then enter the number of persons and their names.
    Number of persons: 1
    0 - name: Robert 
5. Follow the instruction from the Vision window when it appears.
   Please, press enter when
   Robert
   will look to the camera.
6. Press enter and wait 'till the script collects information about the person.
7. You are done! The script will ask you a group to recognize. You can enter the group (MyGroup1) that you just created.

### How to start recognizing
1. Run python script main.py.
2. Press Enter when the scripts asks "Add a new group to the Visit Control system? y/N:".
3. Enter the name of the group of people you want to recognize when the scripts asks "Recognize group: ".


### Algorithm
1. Train the recognizer using a list of faces and save the result to a .yml file.
2. Recognize faces by the saved .yml file.
3. Decide who was recognized and save result in a .csv file.


### Notes
The results of the script are saved to the visits/ folder in .csv format.

Persons are saved to groups/ folder in .json format when training the recognizer.

Pictures (dataset) are stored in dataset/ folder in .jpg format.

Faces are being detected using the default face haar cascade file.

The .yml files are stored in trainer/ folder.
