from tkinter import * 
from tkinter import filedialog
import os

root = Tk()
root.title('program')
root.geometry('400x400')

def open_program():
	
	os.system('python detectmini.py')

def open_program2():
	
	os.system('python testmini_webcam.py')

my_button = Button(root, text="Facerecognition",fg='black',bg="#AAAAAA",command = open_program
my_button.pack(pady = 50)

my_button2 = Button(root, text="IDCARD",fg='black',bg="#AAAAAA", command = open_program2
my_button2.pack(pady = 50)


root.mainloop()
