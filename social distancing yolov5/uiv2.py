import tkinter as tk
import tkinter.font as tkFont
import os

class App:
    def __init__(self, root):
        #setting title
        root.title("Progarm")
        #setting window size
        width=600
        height=500
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GLabel_890=tk.Label(root)
        ft = tkFont.Font(family='Times',size=30)
        GLabel_890["font"] = ft
        GLabel_890["fg"] = "#333333"
        GLabel_890["justify"] = "center"
        GLabel_890["text"] = "ลงทะเบียน"
        GLabel_890.place(x=180,y=20,width=262,height=81)

        GButton_331=tk.Button(root)
        GButton_331["bg"] = "#ffe294"
        ft = tkFont.Font(family='Times',size=18)
        GButton_331["font"] = ft
        GButton_331["fg"] = "#000000"
        GButton_331["justify"] = "center"
        GButton_331["text"] = "Facerecognition"
        GButton_331.place(x=220,y=200,width=165,height=66)
        GButton_331["command"] = self.Progarm

        GButton_681=tk.Button(root)
        GButton_681["activebackground"] = "#ffffff"
        GButton_681["bg"] = "#f9a7a7"
        ft = tkFont.Font(family='Times',size=18)
        GButton_681["font"] = ft
        GButton_681["fg"] = "#000000"
        GButton_681["justify"] = "center"
        GButton_681["text"] = "IDCARD"
        GButton_681.place(x=220,y=340,width=166,height=66)
        GButton_681["command"] = self.Progarm2

    def Progarm(self):
        
        os.system('python detectmini.py')

    def Progarm2(self):

        os.system('python testmini_webcam.py')

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
