# import the necessary packages
# from imutils import face_utils
import numpy as np
import argparse
# import imutils
import dlib
import cv2
from PIL import Image
import pytesseract
import mysql.connector as con
from mysql.connector import errorcode
import datetime
import os

"เวลาปัจจุบัน"
now = datetime.datetime.now()
# now = now.strftime("%d-%m-%y %H:%M:%S")
now_date = now.strftime("%d-%m-%y")
now_time_colon = now.strftime("%H:%M:%S")
now_time = now.strftime("%H-%M-%S")
now_date_folder = now.strftime("%y-%m-%d")
directory = now_date_folder

"เชื่อมต่อฐานข้อมูล"
def connectDB():
    try:
        hosts = "127.0.0.1"
        username = "root"
        passwords = ""
        database_name = "senior_project"
        ports = 3306
        connect = con.connect(user=username, password=passwords, host=hosts, database=database_name, port=ports)
        return connect

    except con.Error as err:
        if err.errno == errorcode.ER.ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

"แสดงข้อมูลทั้งหมดในฐานข้อมูล"
def queryData():
    connect = connectDB()
    cursor = connect.cursor()
    sql = "SELECT * FROM 2type_checkin"
    cursor.execute(sql)
    data = cursor.fetchall()
    print(data)

"หาวันที่ว่าวันนี้มีใครเข้ามาบ้าง (DD-MM-YY)"
def date_search(date):
    connect = connectDB()
    cursor = connect.cursor()
    sql = "SELECT * FROM 2type_checkin WHERE date='%s'" % date #DD-MM-YY
    cursor.execute(sql)
    result = cursor.fetchall()

    for x in result:
        print(x)

"หาเลขบัตร+เวลา (HH:MM:SS)"
def idcard_time_search(time):
    connect = connectDB()
    cursor = connect.cursor()
    sql = "SELECT * FROM 2type_checkin WHERE time='%s'" % time
    cursor.execute(sql)
    result =3
    +cursor.fetchall()
    print(result)

date_search("23-05-21")
# print(idcard_time_search("23:02:00"))