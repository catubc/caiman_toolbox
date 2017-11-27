from Tkinter import *
import tkMessageBox
import tkFileDialog 
import numpy as np
from utils import *

def NewFile(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print "New File!... (not implemented)"
    return
    
    Label(root, text="First Name").grid(row=0)
    Label(root, text="Last Name").grid(row=1)

    e1 = Entry(root)
    e2 = Entry(root)

    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)

    print e1.text()

def OpenFile(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print "Open File!... (not implemented)"
    return
    root.minsize(width=800, height=500)
    name = tkFileDialog.askopenfilename(initialdir='/media/cat/')


def Tif_convert(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print "Converting tif to .npy"

    root.minsize(width=800, height=500)
    root.data = emptyObject()
    root.data.root_dir =  '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/'
    root.data.file_name = '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/Registered.tif'

    #******** Select filename:
    def button0():
        print "...selecting file..."
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        #root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)
    
    e1 = Entry(root, justify='left')       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, root.data.file_name)
    e1.grid(row=0, column=1)
    e1.place(x=120,width=800)
   
    def button1():
        print "...converting: ", root.data.file_name
        convert_tif_npy(root.data.file_name)
        #os.system("python ../CaImAn/demo_OnACID.py "+root.data.file_name)
        print "... done!"
        
    #******** Run review ROIs function
    b1 = Button(root, text="convert tif->npy", command=button1)
    b1.grid(row=1, column=0)



def Caiman_online(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    root.minsize(width=800, height=500)
    root.data = emptyObject()
    root.data.root_dir =  '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/'
    root.data.file_name = '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/Registered.tif'

    #******** Select filename:
    def button0():
        print "...selecting file..."
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename", anchor="w", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)

    e1 = Entry(root, justify='left')       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, root.data.file_name)
    e1.grid(row=0, column=1)
    e1.place(x=120,width=800)
    
    def button1():
        print root.data.file_name
        os.system("python ../CaImAn/demo_OnACID.py "+root.data.file_name)
        
    #******** Run review ROIs function
    b1 = Button(root, text="demo_OnACID", command=button1)
    b1.grid(row=1, column=0)

    
def Caiman_offline(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print "...caiman offline..."


def Image_registration(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()
    print "...image registration ..."


class emptyObject():
    def __init__(self):
        pass
       
        
def Review_ROIs(root):
    print "...Review ROIs ..."
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()
    
    root.data = emptyObject()
    root.data.root_dir = '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/'
    root.data.file_name = '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/G2M5_C1V1_GCaMP6s_20170511_000.npz'


    def load_data():
        data_in = np.load(root.data.file_name)
        
        #Update filename box
        e1.delete(0, END)
        e1.insert(0, root.data.file_name)

        root.data.Yr = data_in['Yr']
        #print Yr.shape

        root.data.YrA = data_in['YrA']
        #print YrA.shape

        A = data_in['A']        #Convert array from sparse to dense
        root.data.A = A[()].toarray()
        #print A.shape

        root.data.C = data_in['C']
        root.data.b = data_in['b']
        root.data.f = data_in['f']
        root.data.Cn = data_in['Cn']

        save_traces(root.data.file_name, root.data.Yr, root.data.A, root.data.C, root.data.b, root.data.f, 250, 250, YrA = root.data.YrA, thr = 0.8, 
            image_neurons=root.data.Cn, denoised_color='red')

    def button0():
        print "...selecting file..."
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        root.title(os.path.split(root.data.file_name)[1])
        
        load_data()
            
    def button1():
        print "... running correct ROIs..."
        correct_ROIs(root.data.file_name, root.data.A, root.data.Cn, thr=0.95)

    def button2():
        print "...plotting contours..."
        plot_contours(root.data.A, root.data.Cn, thr=0.9)

    def button3():
        print "... view patches ..."
        nb_view_patches(root.data.file_name, root.data.Yr, root.data.A, root.data.C, root.data.b, root.data.f, 250, 250, YrA = root.data.YrA, thr = 0.8, 
            image_neurons=root.data.Cn, denoised_color='red')
    
   
    #******** Select filename:
    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)

    e1 = Entry(root)        #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, root.data.file_name)
    e1.grid(row=0, column=1)
    e1.place(x=120,width=800)


    #******** Run review ROIs function
    b1 = Button(root, text="Review ROIs", command=button1)
    b1.grid(row=1, column=0)


    #******** Run review ROIs function
    b2 = Button(root, text="Plot contours", command=button2)
    b2.grid(row=2, column=0)

    #******** Run review ROIs function
    b3 = Button(root, text="View patches", command=button3)
    b3.grid(row=3, column=0)


    load_data()
    #b2 = Button(root, text="correct ROIs ", command=button2)
    #b2.pack()


    #np.save(file_name[:-4]+"_yarray", Yr)

