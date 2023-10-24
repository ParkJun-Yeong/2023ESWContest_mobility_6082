import serial
import json
import pymysql
from datetime import datetime

ser = serial.Serial('/dev/ttyACM0', 9600)

def read_json():
    if ser.in_waiting > 0:
        # print(ser.in_waiting)
        json_str = ser.read(ser.in_waiting).decode()
        data = json.loads(json_str)
        return data
i = 0

while True:
    i += 1
    print(i)
    received_data = read_json()
    if received_data is None:
        continue
    print(received_data)
    type = received_data['type']
    joint = received_data['joint']
    anomal = received_data['anomal']

    try:
        timestamp = received_data['timestamp']
    except:
        timestamp = datetime.now()

    name = "JunYeong"
    email = "wndwls1024@naver.com"
    sql = "INSERT INTO `Walkure_log`(`TYPE`, `JOINT`, `ANOMAL`, `TIMESTAMP`) VALUES (%s,%s,%s,%s)"

    now = datetime.now()
    conn = pymysql.connect(host='mlpalab.synology.me',
                        user='juny',
                        password='Young@62482508',
                        db='juny',
                        port=3306,
                        charset='utf8')

    with conn:
        with conn.cursor() as cur:
            cur.execute(sql, (type, str(';'.join([str(j) for j in joint])), anomal, timestamp))
            conn.commit()
 

# while(True):
#     try:
#         received_data = read_json()

#         conn = pymysql.connect(host='mlpalab.synology.me',
#                                user='juny'
#                                password='Young@62482508',
#                                db='juny',
#                                port='3306'
#                                charset='utf8')
        
#         sql = "INSERT INTO user (name, email) VALUES (%s, %s)"

#         with conn:
#             with conn.cursor() as cur:
#                 cur.execute(sql)

#         print(received_data)
#     except:
#         continue