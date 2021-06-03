from datetime import datetime
import recognition


face_recognition = recognition.FaceRecognition()
add_persons = input('Add a new group to the Visit Control system? y/N:')

# add a new group of people
if add_persons == 'y' or add_persons == 'Y':
    group_name = input('New group: ')
    count = int(input('Number of persons: '))
    persons = [{i: input(str(i)+" - name: ")} for i in range(count)]
    face_recognition.add_group(persons, group_name)

# start recognizing
group_name = input('Recognize group: ')
result, persons = face_recognition.recognize(group_name)

# save visitors to a csv file
visitors = [['id', 'name', 'recognized']]
for id, name in persons.items():
    visitors.append([id, name, int(id) in result])
filename = str(datetime.now())
face_recognition.save_list_csv(visitors, filename)

print('Saved as ' + filename + '.csv')