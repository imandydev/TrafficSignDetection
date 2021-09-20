import tkinter as tk

import tkinter.font as font
import run_camera_cnn as camera_cnn
import run_camera_svm as camera_svm


root = tk.Tk()

root.resizable(False, False)
root.title("Menu")

# căn giữa
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (600/2))
y_cordinate = int((screen_height/2) - (500/2))
root.geometry("{}x{}+{}+{}".format(600, 500, x_cordinate, y_cordinate))


photo = tk.PhotoImage(file='images_root/8.png')
img_label = tk.Label(root,image=photo)
img_label.pack()

myFont = font.Font(weight="bold",size=25)

btnSVM = tk.Button(root, text="SVM", width=10, background="#FF2225", fg="white" , font=myFont, command=camera_svm.runCamera)
btnSVM.pack()
btnSVM.place(x=200, y=160)

btnCNN = tk.Button(root, text="CNN", width=10, background="#FF2225", fg="white", font=myFont, command=camera_cnn.runCamera)
btnCNN.pack()
btnCNN.place(x=200, y=290)

root.mainloop()