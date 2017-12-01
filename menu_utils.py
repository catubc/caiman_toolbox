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
        #root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension=".tif", filetypes=(("tif", "*.tif"),("All Files", "*.*") ))

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
    
    #print '...text************'

    root.minsize(width=800, height=600)
    root.data = emptyObject()
    #root.data_folder root.data.root_dir =  '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/'
    root.data.file_name = ''

    #root.caiman_folder = np.loadtxt('caiman_folder_location.txt',dtype=str)


    #******** Select CaImAn folder
    def button00():
        print "...selecting caiman folder location..."
        root.caiman_folder = tkFileDialog.askdirectory(initialdir=root.caiman_folder, title="Select CaImAn Root Directory")
        print "Changing caiman_folder to: ", root.caiman_folder
        np.savetxt('caiman_folder_location.txt',[root.caiman_folder], fmt="%s") 
        e0.delete(0, END)
        e0.insert(0, root.caiman_folder)
        
    b00 = Button(root, text="CaImAn Folder", anchor="w", command=button00) #Label(root, text="Filename: ").grid(row=0)
    b00.place(x=0,y=0)

    e0 = Entry(root, justify='left')       #text entry for the filename
    e0.delete(0, END)
    e0.insert(0, root.caiman_folder)
    e0.place(x=120,y=0, width=600)


    #******** Filename Selector
    def button0():
        print "...selecting file..."
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension=".tif", filetypes=(("tif", "*.tif"),("npy", "*.npy"),("All Files", "*.*") ))

        print root.data.file_name
        root.data_folder = os.path.split(root.data.file_name)[0]
        np.savetxt('data_folder_location.txt',[root.data_folder], fmt="%s") 
        e.delete(0, END)
        e.insert(0, root.data.file_name)
        root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename:", anchor="w", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.place(x=0, y=30)

    e = Entry(root, justify='left')       #text entry for the filename
    e.delete(0, END)
    e.insert(0, root.data.file_name)
    e.place(x=120,y=30, width=600)


    #******** DEMO_ONACID PARAMETERS ******************
    #Param 1
    x_offset = 10; y_offset=75
    l1 = Label(root, text='Merge Threshold')
    l1.place(x=x_offset,y=y_offset, height=30,width=100)
    
    e1 = Entry(root, justify='left', width=4)       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, 0.8)
    e1.place(x=x_offset+103,y=y_offset+5)
    x_offset+=140
    
    #Param 2
    l2 = Label(root, text='Autoregress order')
    l2.place(x=x_offset,y=y_offset, height=30,width=130)
    
    e2 = Entry(root, justify='left', width=3)       #text entry for the filename
    e2.delete(0, END)
    e2.insert(0, 1)
    e2.place(x=x_offset+120,y=y_offset+5)
    x_offset+=150

    #Param 3
    l3 = Label(root, text='Initial Batch')
    l3.place(x=x_offset,y=y_offset, height=30,width=100)
    
    e3 = Entry(root, justify='left', width=5)       #text entry for the filename
    e3.delete(0, END)
    e3.insert(0, 20000)
    e3.place(x=x_offset+88,y=y_offset+5)
    x_offset+=145

    #Param 4
    l4 = Label(root, text='rf')
    l4.place(x=x_offset,y=y_offset, height=30,width=25)
    
    e4 = Entry(root, justify='left', width=3)       #text entry for the filename
    e4.delete(0, END)
    e4.insert(0, 16)
    e4.place(x=x_offset+22,y=y_offset+5)
    x_offset+=60
    
    #Param 5
    l5 = Label(root, text='stride')
    l5.place(x=x_offset,y=y_offset, height=30,width=40)
    
    e5 = Entry(root, justify='left', width=3)       #text entry for the filename
    e5.delete(0, END)
    e5.insert(0, 3)
    e5.place(x=x_offset+38,y=y_offset+5)
    x_offset+=100

    #******************************* NEW LINE ************************
    x_offset=10; y_offset=y_offset+30

    l6 = Label(root, text='K')
    l6.place(x=x_offset,y=y_offset, height=30,width=15)
    
    e6 = Entry(root, justify='left', width=3)       #text entry for the filename
    e6.delete(0, END)
    e6.insert(0, 4)
    e6.place(x=x_offset+15,y=y_offset+5)
    x_offset+=60
    
    #Param 2
    l7 = Label(root, text='gSig')
    l7.place(x=x_offset,y=y_offset, height=30,width=40)
    
    e7 = Entry(root, justify='left', width=4)       #text entry for the filename
    e7.delete(0, END)
    e7.insert(0, '6, 6')
    e7.place(x=x_offset+40,y=y_offset+5)
    x_offset+=80

    #Param 3
    l8 = Label(root, text='rval_thr')
    l8.place(x=x_offset,y=y_offset, height=30,width=80)
    
    e8 = Entry(root, justify='left', width=5)       #text entry for the filename
    e8.delete(0, END)
    e8.insert(0, 0.95)
    e8.place(x=x_offset+67,y=y_offset+5)
    x_offset+=125

    #Param 4
    l9 = Label(root, text='thr_fitness_delta')
    l9.place(x=x_offset,y=y_offset, height=30,width=110)
    
    e9 = Entry(root, justify='left', width=3)       #text entry for the filename
    e9.delete(0, END)
    e9.insert(0, -50)
    e9.place(x=x_offset+108,y=y_offset+5)
    x_offset+=160
    
    #Param 5
    l10 = Label(root, text='thr_fitness_raw')
    l10.place(x=x_offset,y=y_offset, height=30,width=110)
    
    e10 = Entry(root, justify='left', width=3)       #text entry for the filename
    e10.delete(0, END)
    e10.insert(0, -50)
    e10.place(x=x_offset+106,y=y_offset+5)
   
        
    #********** COMMAND LINE OUTPUT BOX **********
    tkinter_window = False       #Redirect command line outputs to text box in tkinter;
    if tkinter_window:
        t = Text(root, wrap='word', height = 20, width=100)
        t.place(x=10, y=250, in_=root)

    #********* DEMO_ONACID BUTTON **********************
    def button1(l):
        l.config(foreground='red')
        root.update()

        if tkinter_window:
            import io, subprocess
            #proc = subprocess.Popen(["python", "-u", "/home/cat/code/CaImAn/demo_OnACID.py", root.data.file_name], stdout=subprocess.PIPE)
            proc = subprocess.Popen(["python", "-u", "/home/cat/code/CaImAn/demo_OnACID_2.py", root.data.file_name], stdout=subprocess.PIPE)

            while True:
              line = proc.stdout.readline()
              if line != '':
                t.insert(END, '%s\n' % line.rstrip())
                t.see(END)
                t.update_idletasks()
                sys.stdout.flush()
              else:
                break
        else:
            print "python -u "
            print root.caiman_folder
            p = os.system("python -u "+str(root.caiman_folder)+"/demo_OnACID_2.py "+root.data.file_name)
        
    l = Label(root, textvariable='green', fg = 'red')
    b1 = Button(root, text="demo_OnACID", foreground='blue', command=lambda: button1(l))
    b1.place(x=10, y=150, in_=root)

    
    
def Caiman_offline(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print "...caiman offline...(not implemented)"


def Image_registration(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()
    print "...image registration ...(not implemented)"


class emptyObject():
    def __init__(self):
        pass

        
def Review_ROIs(root):
    print "...Review ROIs ..."
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()
    
    root.data = emptyObject()
    root.data.file_name = ''
    root.title("Review and Cleanup ROIs")

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
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension="*processed.npz", filetypes=(("npz", "*processed.npz"),("All Files", "*.*") ))

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
    e1.place(x=220,width=800)


    #******** Run review ROIs function
    b1 = Button(root, text="Review ROIs", command=button1, justify='left')
    b1.place(x=0, y=50)

    #******** Run review ROIs function
    b2 = Button(root, text="Plot contours", command=button2, justify='left')
    b2.place(x=0, y=80)

    #******** Run review ROIs function
    b3 = Button(root, text="View patches", command=button3, justify='left')
    b3.place(x=0, y=110)



def Review_spikes(root):
    print "...Review spikes ..."
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    root.title("Review Spikes")
    root.data = emptyObject()
    root.data.file_name = ''

    #******** Select ROI filename
    def button0():
        print "...selecting ROI file..."
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension="ROIs.npz", filetypes=(("ROI", "*ROIs.npz"),("All Files", "*.*") ))
        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        
        #load_data()

    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)

    e1 = Entry(root)        #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, root.data.file_name)
    e1.place(x=120,width=800)


    #********* Run foopsi
    #Param 1
    x_offset = 0; y_offset=50
    l1 = Label(root, text='Foopsi Threshold')
    l1.place(x=x_offset,y=y_offset, height=30,width=110)
    
    e2 = Entry(root, justify='left', width=4)       #text entry for the filename
    e2.delete(0, END)
    e2.insert(0, 2.0)
    e2.place(x=x_offset+125,y=y_offset+5)
    
    def button1():
        print "...running foopsi..."
        root.data.foopsi_threshold = e2.get()
        run_foopsi(root)

    b1 = Button(root, text="Run Foopsi", command=button1)
    b1.place(x=0,y=80)
    

    #********* View rasters
    def button2():
        print "...viewing rasters..."
        view_rasters(root)

    b2 = Button(root, text="View rasters", command=button2)
    b2.place(x=0,y=150)
            
