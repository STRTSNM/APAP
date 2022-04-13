from pyexpat import model
from sqlite3 import complete_statement
from ssl import OP_NO_RENEGOTIATION
from syslog import LOG_EMERG
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

import os

df = pd.read_csv('diseasesdb.csv')

x = df.drop(['probstat'], axis='columns')
y = df.probstat

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=1)

model_layer1 = SVC(gamma=10, kernel='poly')
model_layer1.fit(x_train, y_train)
score = model_layer1.score(x_train, y_train)
print("model score :", score)

patientdat = open('patients.dat', 'r')

patientDatList = []
patient_result_0 = []
patient_result_1 = []

patient_ni = []
patient_i = []
patient_vi = []

patient_ni_id = []
patient_i_id = []
patient_vi_id = []

patient_ni_value = []
patient_i_value = []
patient_vi_value = []

patient_ni_value_1 = []
patient_i_value_1 = []
patient_vi_value_1 = []

patient_ni_list_id = []
patient_i_list_id = []
patient_vi_list_id = []

for line in patientdat:
    li = list(line.split(" "))
    li.remove("Patient")
    li.remove(":")
    del li[0]
    li[0] = li[0].replace("\n", "")
    patientDatList.append(li)
    print(li)

print(patientDatList)

for i in patientDatList:
    ##print(i)
    element = i[0]
    ##print(element)
    ##print(element, type(element))
    list_for_element = list(element.split(","))
    ##print(list_for_element)
    for e in list_for_element:
        e = float(e)
        ##print(type(e))
    patient_result_0.append(model_layer1.predict([list_for_element[0:]]))

for i in patient_result_0:
    patient_result_1.append(i[0])


print(patient_result_1)

for i, value in enumerate(patient_result_1):
    if value == 1:
        patient_ni.append(patientDatList[i])
    elif value == 2:
        patient_i.append(patientDatList[i])
    elif value == 3:
        patient_vi.append(patientDatList[i])

print("ni :", patient_ni)
print("i :", patient_i)
print("vi :", patient_vi)


############################################################################################################################################################################
#                                                                      layer1 complete                                                                                     #
############################################################################################################################################################################


df = pd.read_csv('diseaseclassi5.csv')

x = df.drop(['probstat'], axis='columns')
y = df.probstat

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=1)
modell2 = LinearRegression()
modell2.fit(x_train, y_train)
score = modell2.score(x_train, y_train)

print("2nd layer model score :", score)


### ni category ###

for i, value in enumerate(patient_ni):
    for ii, val in enumerate(patientDatList):
        if value == val:
            patient_ni_id.append(ii)

print("ni id :", patient_ni_id)


### i category ###

for i, value in enumerate(patient_i):
    for ii, val in enumerate(patientDatList):
        if value == val:
            patient_i_id.append(ii)

print("i id :", patient_i_id)

### vi category ###

for i, value in enumerate(patient_vi):
    for ii, val in enumerate(patientDatList):
        if value == val:
            patient_vi_id.append(ii)

print("vi id :", patient_vi_id)


### ni category prediction ###

for i in range(len(patient_ni_id)):
    st = patientDatList[patient_ni_id[i]]
    element = st[0]
    list_for_element = list(element.split(","))
    ##print(list_for_element)
    for e in list_for_element:
        e = float(e)
    result = modell2.predict([list_for_element[0:]])
    patient_ni_value.append(result)

for i in patient_ni_value:
    patient_ni_value_1.append(i[0])

print(patient_ni_value_1)

patient_ni_value_1_copy = patient_ni_value_1

for i in range(len(patient_ni_value_1)):
    i_max = max(patient_ni_value_1)
    i_max_index = patient_ni_value_1.index(i_max)
    patient_ni_list_id.append(i_max_index)
    patient_ni_value_1[i_max_index] = 0
print(patient_ni_list_id)

patient_ni_final_list = []

for i in range(len(patient_ni_list_id)):
    patient_ni_final_list.append(patientDatList[patient_ni_list_id[i]])

ordered_index_ni = []
for i in range(len(patient_ni_list_id)):
    val = patient_ni_list_id[i]
    id = patient_ni_id[val]
    ordered_index_ni.append(id)
print(ordered_index_ni)


### i category prediction ###


for i in range(len(patient_i_id)):
    st = patientDatList[patient_i_id[i]]
    element = st[0]
    list_for_element = list(element.split(","))
    ##print(list_for_element)
    for e in list_for_element:
        e = float(e)
    result = modell2.predict([list_for_element[0:]])
    patient_i_value.append(result)

for i in patient_i_value:
    patient_i_value_1.append(i[0])

print(patient_i_value_1)

patient_i_value_1_copy = patient_i_value_1

for i in range(len(patient_i_value_1)):
    i_max = max(patient_i_value_1)
    i_max_index = patient_i_value_1.index(i_max)
    patient_i_list_id.append(i_max_index)
    patient_i_value_1[i_max_index] = 0
print(patient_i_list_id)

patient_i_final_list = []

for i in range(len(patient_i_list_id)):
    patient_i_final_list.append(patientDatList[patient_i_list_id[i]])

ordered_index_i = []
for i in range(len(patient_i_list_id)):
    val = patient_i_list_id[i]
    id = patient_i_id[val]
    ordered_index_i.append(id)
print(ordered_index_i)


### vi category prediction ###

for i in range(len(patient_vi_id)):
    st = patientDatList[patient_vi_id[i]]
    element = st[0]
    list_for_element = list(element.split(","))
    ##print(list_for_element)
    for e in list_for_element:
        e = float(e)
    result = modell2.predict([list_for_element[0:]])
    patient_vi_value.append(result)

for i in patient_vi_value:
    patient_vi_value_1.append(i[0])

print(patient_vi_value_1)

patient_vi_value_1_copy = patient_vi_value_1

for i in range(len(patient_vi_value_1)):
    i_max = max(patient_vi_value_1)
    i_max_index = patient_vi_value_1.index(i_max)
    patient_vi_list_id.append(i_max_index)
    patient_vi_value_1[i_max_index] = 0
print(patient_vi_list_id)

patient_vi_final_list = []

for i in range(len(patient_vi_list_id)):
    patient_vi_final_list.append(patientDatList[patient_vi_list_id[i]])

ordered_index_vi = []
for i in range(len(patient_vi_list_id)):
    val = patient_vi_list_id[i]
    id = patient_vi_id[val]
    ordered_index_vi.append(id)
print(ordered_index_vi)

### Final Touch ###

ordered_index_vi.extend(ordered_index_i)
ordered_index_vi.extend(ordered_index_ni)

##os.system("clear")

print("Final list = " ,ordered_index_vi)

for i in ordered_index_vi:
    index = ordered_index_vi.index(i)
    print("Patient ", int(ordered_index_vi[index])+1)