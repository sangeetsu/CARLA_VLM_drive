import pandas as pd
import json
import os

path = 'JSONBASE/'

all_data = []

for filename in os.listdir(path):
    file_path = os.path.join(path,filename)
    with open(file_path, 'r') as json_file:
        ID = filename.split('_')[0]
        data = json.load(json_file)
        throttle = [data['throttle_brake']['kp'], data['throttle_brake']['ki'], data['throttle_brake']['kd']]
        steering = [data['steering']['kp'], data['steering']['ki'], data['steering']['kd']]
        safety = data['safety_buffer']
        speed = data['speed_adhere']
        test = [ID, throttle[0], throttle[1], throttle[2], steering[0], steering[1], steering[2], safety, speed]
        all_data.append(test)

df = pd.DataFrame(all_data)
headers = ["ID","Throttle KP","Throttle KI","Throttle KD","Steering KP","Steering KI","Steering KD","Safety Buffer","Speed Adherence"]
df.columns = headers
df = df.sort_values(by='ID')
df.to_csv('GAPIDFINDINGS.csv',index=False)
print("Good LAWD have mercy")
