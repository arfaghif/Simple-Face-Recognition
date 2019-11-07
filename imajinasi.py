from tkinter import *
from PIL import ImageTk
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import scipy
#from scipy.misc import imread
import _pickle as pickle
#import random
import os
#import matplotlib.pyplot as plt
import imageio
import math
from functools import partial
#from tkinter import ttk

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imageio.imread(image_path, pilmode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error:', e)
        return None

    return dsc


def batch_extractor(images_path, pickled_db_path="create.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        #print ('Extracting features from image %s' % f)
        name = f
        result[name] = extract_features(f)
    w = tk.Label(root, text="done extract",height=2, width= 70)
    w.pack()
    w.place(x=288, y=110)
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)



class Matcher(object):

    def __init__(self, pickled_db_path="create.pck"):
        with open(pickled_db_path,'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        #print(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector, Euclidean):
        # getting cosine distance between search image and images database
        #print(vector)
        #v = vector.reshape(1, -1)
        v = vector.tolist()
        #print(v)
        #m = scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)
        #print(m)
        a =[]
        if (Euclidean):
            for i in range (len(self.matrix)):
                res = dstnc(v, self.matrix[i].tolist())
                a.append(res)
            a = np.array(a)
            return a
        else:
            for i in range (len(self.matrix)):
                res = dotproduct(v,self.matrix[i].tolist())/(lngth(v)*lngth(self.matrix[i].tolist()))
                a.append(res)
            a = np.array(a)
            return a


    def match(self, image_path, topn, Euclidean):
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features, Euclidean)
        #print(img_distances)
        # getting top 5 records
        if (Euclidean):
            nearest_ids = np.argsort(img_distances)[:topn].tolist()
        else:
            nearest_ids = np.argsort(img_distances)[::-1][:topn].tolist()
        
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()


def show_img(path):
    img = imageio.imread(path, pilmode="RGB")
    plt.imshow(img)
    plt.show()


def dstnc(v1, v2):
    sum = 0
    for i in range (len(v1)):
        sum += (v1[i]-v2[i])**2
    return math.sqrt(sum)

def lngth(v1):
    sum = 0
    for i in range (len(v1)):
        sum += v1[i]**2
    return math.sqrt(sum)

def dotproduct(v1, v2):
    sum =0
    for i in range (len(v1)):
        sum+= v1[i] * v2[i]
    return sum

def data():
    for i in range(50):
       Label(frame,text="").grid(row=i,column=0)
       Label(frame,text="        ").grid(row=i,column=1)
       Label(frame,text="        ").grid(row=i,column=2)

def myfunction(event):
    canvas.configure(scrollregion=canvas.bbox("all"),width=250,height=700)

def browse(name):
    browsebutton = Button(root, text="Browse", command= partial(browsefunc,v1.get()==1,name), height=1, width= 68)
    browsebutton.pack(fill=tk.X, pady=5)
    browsebutton.place(x=290, y=110)
#fungsi di panggil di browsebutton nampil image
def browsefunc(n, name):
    global img
    filename = filedialog.askopenfilename()
    #pathlabel.config(text=filename)
    #path.set(filename)
    #img = ImageTk.PhotoImage(Image.open(filename).resize((300,300), Image.ANTIALIAS))
    #a = Label(frame2, image=img)
    #a.grid(row=53, column=1,sticky="ew")
    #img.set(img)
    img = ImageTk.PhotoImage(Image.open(filename).resize((250,250), Image.ANTIALIAS))
    a = Label(root, image=img)
    a.pack()
    a.place(x=400, y =170)
    """b=IntVar()
    b=v.get()"""
    w = tk.Label(root, text="Number of closeness:",height=2, width= 30, justify =LEFT)
    w.pack()
    w.place(x=290, y=470)
    global c
    c = StringVar()
    entry = Entry(root, textvar= c)
    entry.pack()
    entry.place(x=330, y=504)
    
    #run(name,filename, n)
    startbutton = Button(root, text="Start Matching", command= partial(run,name,filename,n), height=1, width= 40)
    startbutton.pack(fill=tk.X, pady=5)
    startbutton.place(x=480, y=500)
    
def run(features, path, n):
    #images_path = 'pins-face-recognition\PINS\pins_Aaron Paul/'
    ma= Matcher (features)
    names, match = ma.match(path, int(c.get()), n)

    global hasil
    hasil=[]
    for i in range(int(c.get())):
        hasil.append(ImageTk.PhotoImage(Image.open(names[i]).resize((150,150), Image.ANTIALIAS)))
        #print(os.path.join(images_path, names[i]))
        a = Label(frame, image=hasil[len(hasil)-1])
        a.grid(row=i, column=1,sticky="ew")

def radio(name):
    global v1
    w = tk.Label(root, text="Choose method:",height=2, width= 30, justify =LEFT)
    w.pack()
    w.place(x=290, y=55)
    v1 = IntVar()
    r1= Radiobutton(root, text='Euclidean Dintance', variable=v1, value=1, command=partial(browse,name))
    r1.pack(anchor=W)
    r1.place(x=500, y=60)
    r2=Radiobutton(root, text='Cosine Simiilarity', variable=v1, value=2, command=partial(browse,name))
    r2.pack(anchor=W)
    r2.place(x=660, y=60)


def extract():
    browsebutton = Button(root, text="Browse", command= partial(browsefunc1,1==v.get()), height=1, width= 68)
    browsebutton.pack(fill=tk.X, pady=5)
    browsebutton.place(x=290, y=110)

def browsefunc1(n):
    if (n):
        w = tk.Label(root, text="Wait extracting process",height=2, width= 70)
        w.pack()
        w.place(x=288, y=110)
        foldername = filedialog.askdirectory()
        files = [os.path.join(foldername, p) for p in sorted(os.listdir(foldername))]
        batch_extractor(foldername)
        radio("create.pck")
        
    else :
        filename = filedialog.askopenfilename()
        radio(filename)

root=Tk()
sizex = 800
sizey = 600
posx  = 0
posy  = 0

#main
root.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))

#radio
w = tk.Label(root, text="Choose Database method:",height=2, width= 30, justify=LEFT)
w.pack()
w.place(x=290, y=15)
v = IntVar()
r1= Radiobutton(root, text='Extract from directory', variable=v, value=1, command=extract)
r1.pack(anchor=W)
r1.place(x=500, y=20)
r2=Radiobutton(root, text='Use file .pck', variable=v, value=2, command=extract)
r2.pack(anchor=W)
r2.place(x=660, y=20)

#browse-button
"""browsebutton = Button(root, text="Browse", command= partial(browsefunc,n), height=1, width= 68)
browsebutton.pack(fill=tk.X, pady=5)
browsebutton.place(x=290, y=70)"""
#frame-inner in canvas 
myframe=Frame(root,relief=GROOVE,width=200,height=100,bd=1)
myframe.place(x=0,y=0)
canvas=Canvas(myframe)
frame=Frame(canvas)

#scrollbar
myscrollbar=Scrollbar(myframe,orient="vertical",command=canvas.yview)
canvas.configure(yscrollcommand=myscrollbar.set)

#scrollbar position
myscrollbar.pack(side="right",fill="y")
canvas.pack(side="left")
canvas.create_window((0,0),window=frame,anchor='nw')
frame.bind("<Configure>",myfunction)



data()

root.mainloop()
"""
canvas1 = Canvas(root, width = 300, height = 300)
canvas1.pack()  
img = ImageTk.PhotoImage(Image.open("Aaron Paul129_259.jpg").resize((200,200), Image.ANTIALIAS))
canvas1.create_image(1000, 10000, anchor=NW, image=img)
canvas1.place(x=200, y=300)"""

