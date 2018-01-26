from tkinter import *
from tkinter import filedialog as tkFileDialog
import numpy as np
from utils import *


def NewFile(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print ("New File!... (not implemented)")
    return
    
    Label(root, text="First Name").grid(row=0)
    Label(root, text="Last Name").grid(row=1)

    e1 = Entry(root)
    e2 = Entry(root)

    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)

    print (e1.text())


def Defaults(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    #******** Select CaImAn folder
    def button00():
        print ("...selecting caiman folder location...")
        root.caiman_folder = tkFileDialog.askdirectory(initialdir=root.caiman_folder, title="Select CaImAn Root Directory")
        print ("Changing caiman_folder to: ", root.caiman_folder)
        np.savetxt('caiman_folder_location.txt',[root.caiman_folder], fmt="%s") 
        e0.delete(0, END)
        e0.insert(0, root.caiman_folder)
        
    b00 = Button(root, text="Set CaImAn Folder", anchor="w", command=button00) #Label(root, text="Filename: ").grid(row=0)
    b00.place(x=0,y=0)

    e0 = Entry(root, justify='left')       #text entry for the filename
    e0.delete(0, END)
    e0.insert(0, root.caiman_folder)
    e0.place(x=150,y=0, width=600)


    #******** Filename Selector
    def button0():
        print ("...selecting data folder...")
        root.data_folder = tkFileDialog.askdirectory(initialdir=root.data_folder, title="Select data directory")
        np.savetxt('data_folder_location.txt',[root.data_folder], fmt="%s") 
        e.delete(0, END)
        e.insert(0, root.data_folder)
        #root.title(root.data_folder)
        
    b0 = Button(root, text="Default data dir:", anchor="w", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.place(x=0, y=30)

    e = Entry(root, justify='left')       #text entry for the filename
    e.delete(0, END)
    e.insert(0, root.data_folder)
    e.place(x=150,y=30, width=600)

def Tif_merge(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print ("Merging tifs from list of filenames")

    root.minsize(width=800, height=500)
    root.data = emptyObject()

    #******** Select filename:
    def button0():
        print ("...selecting .txt containing list of .tif files...")
        #root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension=".txt", filetypes=(("txt", "*.txt"),("All Files", "*.*") ))

        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        #root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)
    
    e1 = Entry(root, justify='left')       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, '')
    e1.grid(row=0, column=1)
    e1.place(x=120,width=800)
    
    def button1():
        print ("...merging files from list: ", root.data.file_name)
        merge_tifs(root.data.file_name)
        #os.system("python ../CaImAn/demo_OnACID.py "+root.data.file_name)
        print ("... done!")
        
    #******** Run review ROIs function
    b1 = Button(root, text="merge tifs", command=button1)
    b1.grid(row=1, column=0)
    
def Tif_convert(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print ("Converting tif to .npy")

    root.minsize(width=800, height=500)
    root.data = emptyObject()
    #root.data.root_dir =  '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/'
    #root.data.file_name = '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/Registered.tif'

    #******** Select filename:
    def button0():
        print ("...selecting file...")
        #root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension=".tif", filetypes=(("tif", "*.tif"),("All Files", "*.*") ))

        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        #root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)
    
    e1 = Entry(root, justify='left')       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, '')
    e1.grid(row=0, column=1)
    e1.place(x=120,width=800)
   
    def button1():
        print ("...converting: ", root.data.file_name)
        convert_tif_npy(root.data.file_name)
        #os.system("python ../CaImAn/demo_OnACID.py "+root.data.file_name)
        print ("... done!")
        
    #******** Run review ROIs function
    b1 = Button(root, text="convert tif->npy", command=button1)
    b1.grid(row=1, column=0)

def Tif_sequence_load(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print ("Loading .tif sequence")
    
    print ("... buggy... not yet fixed...(tiffilfe seems to give errors...)")
    return

    root.minsize(width=1200, height=500)
    root.data = emptyObject()

    #******** Select filename:
    def button0():
        print ("...selecting file...")
        #root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension=".tif", filetypes=(("tif", "*.tif"),("All Files", "*.*") ))

        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        #root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)
    
    e1 = Entry(root, justify='left')       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, '')
    e1.grid(row=0, column=1)
    e1.place(x=120,width=1200)
   
    def button1():
        print (e1.get())
        root.data.file_name = e1.get()

        print ("...loading: ", root.data.file_name)
        load_tif_sequence(e1.get())

        print ("... done!")
        
    #******** Run review ROIs function
    b1 = Button(root, text="Load tif sequence", command=button1)
    b1.grid(row=1, column=0)

def Crop_rectangle(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print ("Croping image")

    root.minsize(width=800, height=500)
    root.data = emptyObject()
    #root.data.root_dir =  '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/'
    #root.data.file_name = '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/Registered.tif'

    #******** Select filename:
    def button0():
        print ("...selecting file...")
        #root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension=".tif", filetypes=(("tif", "*.tif"),("All Files", "*.*") ))

        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        #root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)
    
    e1 = Entry(root, justify='left')       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, '')
    e1.grid(row=0, column=1)
    e1.place(x=120,width=800)
   
    def button1():
        print ("...croping: ", root.data.file_name)
        crop_image(root.data.file_name)
        #os.system("python ../CaImAn/demo_OnACID.py "+root.data.file_name)
        print ("... done!")
        
    #******** Run review ROIs function
    b1 = Button(root, text="Crop tif (or .npy)", command=button1)
    b1.grid(row=1, column=0)

def Crop_arbitrary(root):
    print ("... arbitrary crop not yet implemented ...")

def Caiman_online(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()
    
    #print '...text************'

    root.minsize(width=1000, height=600)
    root.data = emptyObject()
    #root.data_folder root.data.root_dir =  '/media/cat/4TB/in_vivo/rafa/alejandro/G2M5/20170511/000/'
    root.data.file_name = ''

    #root.caiman_folder = np.loadtxt('caiman_folder_location.txt',dtype=str)

    #******** Filename Selector
    def button0():
        print ("...selecting file...")
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension=".tif", filetypes=(("tif", "*.tif"),("npy", "*.npy"),("All Files", "*.*") ))

        print (root.data.file_name)
        root.data_folder = os.path.split(root.data.file_name)[0]
        np.savetxt('data_folder_location.txt',[root.data_folder], fmt="%s") 
        e.delete(0, END)
        e.insert(0, root.data.file_name)
        root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename:", anchor="w", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.place(x=0, y=0)

    e = Entry(root, justify='left')       #text entry for the filename
    e.delete(0, END)
    e.insert(0, root.data.file_name)
    e.place(x=110,y=4, width=600)
    
    x_offset=0; y_offset=30
   
   
    l00 = Label(root, text='_'*200)
    l00.place(x=x_offset, y=y_offset, height=30, width=1000)

   
    #******** CNMF Parameters ******************
    #
    x_offset = 0; y_offset=55
    l0 = Label(root, text='CNMF Initialization Parameters',  fg="red", justify='left')
    l0.place(x=x_offset, y=y_offset, height=30, width=190)
    
    #Param 1
    x_offset=10; y_offset=+80
    l1 = Label(root, text='Merge Threshold')
    l1.place(x=x_offset,y=y_offset, height=30,width=100)
    
    e1 = Entry(root, justify='left', width=4)       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, 0.8)
    e1.place(x=x_offset+103,y=y_offset+5)
    x_offset+=140
    
    ##Param 2
    #l2 = Label(root, text='Autoregress order')
    #l2.place(x=x_offset,y=y_offset, height=30,width=130)
    
    #e2 = Entry(root, justify='left', width=3)       #text entry for the filename
    #e2.delete(0, END)
    #e2.insert(0, 1)
    #e2.place(x=x_offset+120,y=y_offset+5)
    #x_offset+=150

    #Param 3
    l3 = Label(root, text='Initial Batch')
    l3.place(x=x_offset,y=y_offset, height=30,width=100)
    
    e3 = Entry(root, justify='left', width=5)       #text entry for the filename
    e3.delete(0, END)
    e3.insert(0, 20000)
    e3.place(x=x_offset+88,y=y_offset+5)
    x_offset+=145

    #Param 4
    l4 = Label(root, text='patch_size')
    l4.place(x=x_offset,y=y_offset, height=30,width=100)
    
    e4 = Entry(root, justify='left', width=3)       #text entry for the filename
    e4.delete(0, END)
    e4.insert(0, 32)
    e4.place(x=x_offset+85,y=y_offset+5)
    x_offset+=160
    
    #Param 5
    l5 = Label(root, text='stride')
    l5.place(x=x_offset,y=y_offset, height=30,width=40)
    
    e5 = Entry(root, justify='left', width=3)       #text entry for the filename
    e5.delete(0, END)
    e5.insert(0, 3)
    e5.place(x=x_offset+38,y=y_offset+5)
    x_offset+=100
    
    #Param 6
    l6 = Label(root, text='K')
    l6.place(x=x_offset,y=y_offset, height=30,width=15)
    
    e6 = Entry(root, justify='left', width=3)       #text entry for the filename
    e6.delete(0, END)
    e6.insert(0, 4)
    e6.place(x=x_offset+15,y=y_offset+5)
    
    
    #***************************************************
    #Recording Defaults
    #NEW LINE 
    x_offset = 0; y_offset+=50
    print (x_offset, y_offset)
    l_1 = Label(root, text='Recording Defaults',  fg="blue", justify='left')
    l_1.place(x=x_offset, y=y_offset, height=30, width=120)
    
    y_offset+=25    
    #Param 2
    l7 = Label(root, text='frame_rate (hz)')
    l7.place(x=x_offset,y=y_offset, height=30,width=110)
    
    x_offset+=105
    e7 = Entry(root, justify='left', width=4)       #text entry for the filename
    e7.delete(0, END)
    e7.insert(0, 10)
    e7.place(x=x_offset,y=y_offset+5)

    #Param 3
    x_offset+=30
    l8 = Label(root, text='decay_time (s)')
    l8.place(x=x_offset,y=y_offset, height=30,width=110)
    
    x_offset+=100
    e8 = Entry(root, justify='left', width=4)       #text entry for the filename
    e8.delete(0, END)
    e8.insert(0, 0.5)
    e8.place(x=x_offset,y=y_offset+5)

    #Param 3
    x_offset+=50
    l9 = Label(root, text='neuron (pixels)')
    l9.place(x=x_offset,y=y_offset, height=30,width=100)
    
    x_offset+=100
    e9 = Entry(root, justify='left', width=4)       #text entry for the filename
    e9.delete(0, END)
    e9.insert(0, '6, 6')
    e9.place(x=x_offset,y=y_offset+5)


    #Param 3
    x_offset+=40
    l10 = Label(root, text='order AR dynamics')
    l10.place(x=x_offset,y=y_offset, height=30,width=145)
    
    x_offset+=135
    e10 = Entry(root, justify='left', width=3)       #text entry for the filename
    e10.delete(0, END)
    e10.insert(0, 1)
    e10.place(x=x_offset,y=y_offset+5)


    #Param 
    x_offset+=40
    l11 = Label(root, text='min_SNR')
    l11.place(x=x_offset,y=y_offset, height=30,width=65)
    
    x_offset+=62
    e11 = Entry(root, justify='left', width=4)       #text entry for the filename
    e11.delete(0, END)
    e11.insert(0, 3.5)
    e11.place(x=x_offset,y=y_offset+5)


    #Param 
    x_offset+=40
    l12 = Label(root, text='rval_thr')
    l12.place(x=x_offset,y=y_offset, height=30,width=65)
    
    x_offset+=60
    e12 = Entry(root, justify='left', width=4)       #text entry for the filename
    e12.delete(0, END)
    e12.insert(0, 0.90)
    e12.place(x=x_offset,y=y_offset+5)


    #Param 
    x_offset+=40
    l13 = Label(root, text='# bkgr comp')
    l13.place(x=x_offset,y=y_offset, height=30,width=105)
    
    x_offset+=95
    e13 = Entry(root, justify='left', width=4)       #text entry for the filename
    e13.delete(0, END)
    e13.insert(0, 3)
    e13.place(x=x_offset,y=y_offset+5)

    
    #***************************************************
    #Temporary Initalization Defaults
    #NEW LINE 
    x_offset = 0; y_offset+=50
    print (x_offset, y_offset)
    l_1 = Label(root, text='Initialization Defaults',  fg="green", justify='left')
    l_1.place(x=x_offset, y=y_offset, height=30, width=140)
    
    y_offset+=30
    #Param 
    x_offset=0; x_width=120
    l14 = Label(root, text='# updated shapes')
    l14.place(x=x_offset,y=y_offset, height=30,width=x_width)
    
    x_offset+=x_width
    e14 = Entry(root, justify='left', width=4)       #text entry for the filename
    e14.delete(0, END)
    e14.insert(0, 'inf')
    e14.place(x=x_offset,y=y_offset+5)

    #Param 
    x_offset+=45; x_width=125
    l15 = Label(root, text='# expected shapes')
    l15.place(x=x_offset,y=y_offset, height=30,width=x_width)
    
    x_offset+=x_width
    e15 = Entry(root, justify='left', width=4)       #text entry for the filename
    e15.delete(0, END)
    e15.insert(0, 250)
    e15.place(x=x_offset,y=y_offset+5)

    #Param 
    x_offset+=45; x_width=80
    l16 = Label(root, text='# timesteps')
    l16.place(x=x_offset,y=y_offset, height=30,width=x_width)
    
    x_offset+=x_width
    e16 = Entry(root, justify='left', width=4)       #text entry for the filename
    e16.delete(0, END)
    N_samples = np.ceil(float(e7.get())*float(e8.get()))
    e16.insert(0, N_samples)
    e16.place(x=x_offset,y=y_offset+5)

    #Param 
    from scipy.special import log_ndtr
    x_offset+=45; x_width=140
    l17 = Label(root, text='exceptionality thresh')
    l17.place(x=x_offset,y=y_offset, height=30,width=x_width)
    
    x_offset+=x_width
    e17 = Entry(root, justify='left', width=5)       #text entry for the filename
    e17.delete(0, END)
    e17.insert(0, log_ndtr(-float(e11.get()))*N_samples)
    e17.place(x=x_offset,y=y_offset+5)

    #Param 
    x_offset+=55; x_width=105
    l18 = Label(root, text='total len of file')
    l18.place(x=x_offset,y=y_offset, height=30,width=x_width)
    
    x_offset+=x_width
    e18 = Entry(root, justify='left', width=5)       #text entry for the filename
    e18.delete(0, END)
    e18.insert(0, 'all')
    e18.place(x=x_offset,y=y_offset+5)

    y_offset+=30; x_offset=0
    l000 = Label(root, text='_'*200)
    l000.place(x=x_offset, y=y_offset, height=30, width=1000)
   
    #********** COMMAND LINE OUTPUT BOX **********
    tkinter_window = False       #Redirect command line outputs to text box in tkinter;
    if tkinter_window:
        t = Text(root, wrap='word', height = 20, width=100)
        t.place(x=10, y=250, in_=root)

    #********* DEMO_ONACID BUTTON **********************
    def button1(l):
        l.config(foreground='red')
        root.update()

        #Save existing config file
        np.savez(str(root.data.file_name)[:-4]+"_runtime_params", \
            merge_thr=e1.get(),     #merge threshold
            initibatch = e3.get(),     #Initial batch
            patch_size=e4.get(),     #patch size
            stride=e5.get(),     #stride
            K=e6.get(),     #K
            frame_rate=e7.get(),      #frame_rate
            decay_time=e8.get(),     #decay_time
            neuron_size=e9.get(),     #neuron size pixesl
            AR_dynamics=e10.get(),    #AR dynamics order
            min_SNR=e11.get(),  #min_SNR
            rval_threshold = e12.get(),    # rval_threshold
            no_bkgr_components=e13.get(),    # #bkground componenets
            no_updated_shapes=e14.get(),    # #udpated shapes
            no_expected_shapes=e15.get(),    # #expected shapes
            no_timesteps=e16.get(),    # #timesteps
            exceptionality_threshold=e17.get(),    # exceptionatliy threshold
            total_len_file=e18.get(),     # total len of file
            caiman_location = str(root.caiman_folder)
            )
        print (type(str(root.caiman_folder)))
        print (type(root.caiman_folder))
                
        if tkinter_window:
            import io, subprocess
            #proc = subprocess.Popen(["python", "-u", "/home/cat/code/CaImAn/demo_OnACID.py", root.data.file_name], stdout=subprocess.PIPE)
            proc = subprocess.Popen(["python", "-u", str(root.caiman_folder)+"/demo_OnACID_2.py", root.data.file_name], stdout=subprocess.PIPE)

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
            print ("python -u ")
            print (root.caiman_folder)
            #p = os.system("python -u "+str(root.caiman_folder)+"/demo_OnACID_2.py "+root.data.file_name)
            print ("python -u "+str(root.caiman_folder)+"/demo_OnACID_2.py "+str(root.data.file_name))
            p = os.system("python -u "+str(root.caiman_folder)+"/demo_OnACID_2.py "+str(root.data.file_name))
        
    l = Label(root, textvariable='green', fg = 'red')
    b1 = Button(root, text="demo_OnACID", foreground='blue', command=lambda: button1(l))
    b1.place(x=0, y=y_offset+50, in_=root)

    
def Caiman_offline(root):
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print ("...caiman offline...(not implemented)")
    print ("...Note: caiman online is currently running in caiman offline mode... ")


def Image_registration(root):
    
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    print ("... image registration...")
    

    root.minsize(width=800, height=500)
    root.data = emptyObject()
    

    #******** Select filename:
    def button0():
        print ("...selecting file...")
        #root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data.root_dir)
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension=".tif", filetypes=(("tif", "*.tif"),("All Files", "*.*") ))

        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        #root.title(os.path.split(root.data.file_name)[1])
        
    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)
    
    e1 = Entry(root, justify='left')       #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, '')
    e1.grid(row=0, column=1)
    e1.place(x=120,width=800)
   
    def button1():
        print ("...motion correcting: ", root.data.file_name)
        motion_correct_caiman(root)
        
    #******** Run review ROIs function
    b1 = Button(root, text="motion correct", command=button1)
    b1.grid(row=1, column=0)



class emptyObject():
    def __init__(self):
        pass

        
def Review_ROIs(root):
    print ("...Review ROIs ...")
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()
    
    root.data = emptyObject()
    root.data.file_name = ''
    root.title("Review and Cleanup ROIs")

    def load_data():
        print ("...loading processed file: ", root.data.file_name)
        #Update filename box
        e1.delete(0, END)
        e1.insert(0, root.data.file_name)

        #Load data
        data_in = np.load(root.data.file_name, encoding= 'latin1',  mmap_mode='c')
        
        print (data_in.keys())

        A = data_in['A']        #Convert array from sparse to dense
        print (A.shape)
        #print (A[()].shape)
        root.data.A = A[()].toarray()
        #print (root.data.A.shape)

        root.data.Yr = data_in['Yr']
        print (root.data.Yr.shape)

        root.data.YrA = data_in['YrA']
        print (root.data.YrA.shape)

        root.data.C = data_in['C']
        root.data.b = data_in['b']
        root.data.f = data_in['f']
        root.data.Cn = data_in['Cn']

        save_traces(root.data.file_name, root.data.Yr, root.data.A, root.data.C, root.data.b, root.data.f, 250, 250, YrA = root.data.YrA, thr = 0.8, 
            image_neurons=root.data.Cn, denoised_color='red')

    #******** Select filename:
    def button0():
        print ("...selecting file...")
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension="*processed.npz", filetypes=(("npz", "*processed.npz"),("All Files", "*.*") ))

        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        root.title(os.path.split(root.data.file_name)[1])
        
        load_data()

    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)

    e1 = Entry(root)        #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, root.data.file_name)
    e1.grid(row=0, column=1)
    e1.place(x=120,width=800)

            
    #******** Correct ROIs
    def button1():
        print ("... running correct ROIs...")
        correct_ROIs(root.data.file_name, root.data.A, root.data.Cn, thr=0.95)

    b1 = Button(root, text="Review ROIs", command=button1, justify='left')
    b1.place(x=0, y=50)


    #************NOT IMPLEMENTED
    #if False:
        #def button2():
            #print ("...plotting contours...")
            #plot_contours(root.data.A, root.data.Cn, thr=0.9)

        #def button3():
            #print ("... view patches ...")
            #nb_view_patches(root.data.file_name, root.data.Yr, root.data.A, root.data.C, root.data.b, root.data.f, 250, 250, YrA = root.data.YrA, thr = 0.8, 
                #image_neurons=root.data.Cn, denoised_color='red')
        

        ##******** Run review ROIs function
        #b2 = Button(root, text="Plot contours", command=button2, justify='left')
        #b2.place(x=0, y=80)

        ##******** Run review ROIs function
        #b3 = Button(root, text="View patches", command=button3, justify='left')
        #b3.place(x=0, y=110)



def Review_spikes(root):
    print ("...Review spikes ...")
    for k, ele in enumerate(root.winfo_children()):
        if k>0: ele.destroy()

    root.title("Review Spikes")
    root.data = emptyObject()
    root.data.file_name = ''

    #******** Select ROI filename
    def button0():
        print ("...selecting ROI file...")
        root.data.file_name = tkFileDialog.askopenfilename(initialdir=root.data_folder, defaultextension="ROIs.npz", filetypes=(("ROI", "*ROIs.npz"),("All Files", "*.*") ))
        e1.delete(0, END)
        e1.insert(0, root.data.file_name)
        
        #load_data()

    b0 = Button(root, text="Filename: ", command=button0) #Label(root, text="Filename: ").grid(row=0)
    b0.grid(row=0,column=0)

    e1 = Entry(root)        #text entry for the filename
    e1.delete(0, END)
    e1.insert(0, root.data.file_name)
    e1.place(x=120,y=5, width=800)


    #********* Run foopsi
    #Param 1
    x_offset = 0; y_offset=50
    l1 = Label(root, text='Foopsi Threshold')
    l1.place(x=x_offset,y=y_offset, height=30,width=110)
    
    x_offset+=125; y_offset+=5
    e2 = Entry(root, justify='left', width=4)       #text entry for the filename
    e2.delete(0, END)
    e2.insert(0, 2.0)
    e2.place(x=x_offset,y=y_offset)
    
    def button1():
        print ("...running foopsi...")
        root.data.foopsi_threshold = e2.get()
        run_foopsi(root)

    x_offset=0; y_offset=80
    b1 = Button(root, text="Run Foopsi", command=button1)
    b1.place(x=x_offset,y=y_offset)
    

    #********* View rasters
    def button2():
        print ("...viewing rasters...")
        root.data.foopsi_threshold = e2.get()
        view_rasters(root)
    
    y_offset+=70
    b2 = Button(root, text="View all rasters", command=button2)
    b2.place(x=x_offset,y=y_offset)
    

    #********* View single neuron
    def button2():
        print ("...viewing trace...")
        root.data.foopsi_threshold = e2.get()
        root.data.neuron_id = int(e3.get())
        view_neuron(root)

    y_offset+=70
    b2 = Button(root, text="View neuron", command=button2)
    b2.place(x=x_offset,y=y_offset)
    
    #x_offset+=100
    #l2 = Label(root, text='Neuron #:')
    #l2.place(x=x_offset,y=y_offset, height=30,width=110)
    
    x_offset+=125; y_offset+=5
    e3 = Entry(root, justify='left', width=4)       #text entry for the filename
    e3.delete(0, END)
    e3.insert(0, 0)
    e3.place(x=x_offset,y=y_offset)
            
