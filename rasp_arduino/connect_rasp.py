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

    name = ""
    email = ""
    sql = "INSERT INTO `Walkure_log`(`TYPE`, `JOINT`, `ANOMAL`, `TIMESTAMP`) VALUES (%s,%s,%s,%s)"

    now = datetime.now()
    conn = pymysql.connect(host='',
                        user='',
                        password='',
                        db='',
                        port=,
                        charset='utf8')

    with conn:
        with conn.cursor() as cur:
            cur.execute(sql, (type, str(';'.join([str(j) for j in joint])), anomal, timestamp))
            conn.commit()
 
