from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras

sym2intf = open('symbols.csv')
sym2intf.readline()
sym2int = {}
sym2tex = {}
i=0
for line in sym2intf:
    V = line.split(',')
    sym2int[int(V[0])] = i
    sym2tex[int(V[0])] = V[1]
    i = i+1

num_labels = i
int2sym = dict(map(reversed, sym2int.items()))


class gui():

    def __init__(self):

        self.CNN4 = tf.keras.models.load_model('model1')
        self.CNN4.summary()

        self.CNN6 = tf.keras.models.load_model('model2')
        self.CNN6.summary()

        self.CNN8 = tf.keras.models.load_model('model3')
        self.CNN8.summary()

        self.root = Tk()

        self.w = 512
        self.h = 512

        self.c = Canvas(self.root, bg='white', width=self.w, height=self.h)
        self.c.grid(row=1, columnspan=6)

        self.Button1 = Button(self.root, text="Save Image", command=self.savecanvas)
        self.Button1.grid(row=2,column=0)

        self.Button2 = Button(self.root, text="Clear", command=self.clear)
        self.Button2.grid(row=2,column=1)

        self.Button3 = Button(self.root, text="Ausw. CNN 4", command=self.identify4)
        self.Button3.grid(row=2,column=2)
        self.Button4 = Button(self.root, text="Ausw. CNN 6", command=self.identify6)
        self.Button4.grid(row=2,column=3)
        self.Button5 = Button(self.root, text="Ausw. CNN 8", command=self.identify8)
        self.Button5.grid(row=2,column=4)

        self.cc = Image.new("RGB", (self.w, self.h), (255,255,255))
        
        self.c.bind('<B1-Motion>', self.draw)
        self.c.bind('<ButtonRelease-1>', self.release)

        self.old_x = None
        self.old_y = None

        self.root.mainloop()
    
    def draw(self, event):
        self.line_width = 3
        if self.old_x != None and self.old_y != None:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y, width=self.line_width)
            ImageDraw.Draw(self.cc).line([self.old_x, self.old_y, event.x, event.y], (0,0,0), self.line_width)

        self.old_x = event.x
        self.old_y = event.y

    def release(self, event):
        self.old_x = None
        self.old_y = None

    def savecanvas(self):
        print('Saving Image')
        self.cc.save('test.png')

    def clear(self):
        self.c.delete('all')
        self.cc = Image.new("RGB", (self.w, self.h), (255,255,255))

    def identify4(self):
        self.c.delete('token')
        self.identify(self.CNN4)

    def identify6(self):
        self.c.delete('token')
        self.identify(self.CNN6)

    def identify8(self):
        self.c.delete('token')
        self.identify(self.CNN8)

    def identify(self, model):
        img = np.array(self.cc).astype(np.uint8)
        img = cv.blur(img,(4,4))

        imgbw = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        ret, imgt = cv.threshold(imgbw,253,254,cv.THRESH_BINARY_INV)
        num, CC, stats, centroids = cv.connectedComponentsWithStats(imgt.astype(np.uint8),8,cv.CV_32S)

        for j in range(1,num):
            A = (CC == j)
            w = int(max(stats[j][2], stats[j][3])/2)
            x = int(round(centroids[j][1]))
            y = int(round(centroids[j][0]))
            P = A[max(0,x-w):min(self.w,x+w), max(0,y-w):min(y+w,self.h)].astype(np.uint8)
            print(w,x,y)
            resized = 1 - cv.resize(P, (32,32), 0,0,interpolation = cv.INTER_LANCZOS4)
            self.c.create_rectangle(y-w, x-w, y+w, x+w, width=3, outline='red', tags = "token")
            prediction = model.predict(resized.reshape(1,32,32,1))
            res = np.vectorize(sym2tex.get)(np.vectorize(int2sym.get)(np.flip(np.argsort(prediction)))).flatten()
            per = np.flip(np.sort(prediction)).flatten()
            per = per[per>0.1]
            if per.size == 0:
                self.c.create_text(y-w,x-w, anchor=SW, text=f'NICHTS GEFUNDEN', tags = "token")
            elif per.size == 1:
                self.c.create_text(y-w,x-w, anchor=SW, text=f'{res[0]} {per[0]:.2f}', tags = "token")
            elif per.size == 2:
                self.c.create_text(y-w,x-w, anchor=SW, text=f'{res[0]} {per[0]:.2f}, {res[1]} {per[1]:.2f}', tags = "token")
            elif per.size >= 3:
                self.c.create_text(y-w,x-w, anchor=SW, text=f'{res[0]} {per[0]:.2f}, {res[1]} {per[1]:.2f}, {res[2]} {per[2]:.2f}', tags = "token")

gui()