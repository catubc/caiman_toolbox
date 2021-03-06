from tkinter import *

import os
import bokeh
from shutil import copyfile

from menu_utils import  *
    
root = Tk()
menu = Menu(root)
root.config(menu=menu)
root.minsize(width=1200, height=500)
root.title("CaImAn GUI and Analysis Toolbox")
root.geometry('250x150+200+100')
#root.iconbitmap('/home/cat/code/caiman_toolbox/caiman.ico')
#root.wm_iconbitmap('/home/cat/code/caiman_toolbox/caiman.ico')

#imgicon = PhotoImage(file='/home/cat/Downloads/Caiman_logo_FI.png')
#root.tk.call('wm', 'iconphoto', root._w, imgicon)  
#menu.master.iconbitmap('/home/cat/code/caiman_toolbox/caiman.ico')
logo = PhotoImage(file="caiman.gif")
w1 = Label(root, image=logo).pack(side='top')

root.caiman_folder = np.loadtxt('caiman_folder_location.txt',dtype=str)
print ("Location of CaImAn folder: ", str(root.caiman_folder))

root.data_folder = np.loadtxt('data_folder_location.txt',dtype=str)
print ("Location of data folder: ", str(root.data_folder))


#Copying demo_OnACID_2 file to correct location
if os.path.exists(str(root.caiman_folder)+'/demo_OnACID_2.py'):
    pass
else:
    print ("Copying demo_OnACID_2.py file to root caiman_folder")
    try:
        copyfile('demo_OnACID_2.py', str(root.caiman_folder)+'/demo_OnACID_2.py')
    except:
        print ("... incorrect file location..")
#************************************************************
#************************ FILE MENU *************************
#************************************************************
filemenu = Menu(menu)
menu.add_cascade(label="Options", menu=filemenu)
#filemenu.add_command(label="New", command=lambda: NewFile(root))
filemenu.add_command(label="Defaults", command=lambda: Defaults(root))
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

#************************************************************
#*********************** PREPROCESS MENU ********************
#************************************************************
preprocessmenu = Menu(menu)
menu.add_cascade(label="Pre-Process", menu=preprocessmenu)
preprocessmenu.add_command(label="Image Registration", command=lambda: Image_registration(root))
preprocessmenu.add_command(label="Convert .tif -> .npy", command=lambda: Tif_convert(root))
preprocessmenu.add_command(label="Merge .tifs", command=lambda: Tif_merge(root))
preprocessmenu.add_command(label="Load .tif sequence", command=lambda: Tif_sequence_load(root))
preprocessmenu.add_command(label="Rectangle Crop Image", command=lambda: Crop_rectangle(root))
preprocessmenu.add_command(label="Arbitrary Crop Image (not implemented)", command=lambda: Crop_arbitrary(root))

#************************************************************
#*********************** PROCESS MENU ***********************
#************************************************************
processmenu = Menu(menu)
menu.add_cascade(label="Process", menu=processmenu)
processmenu.add_command(label="CaImAn - Online", command=lambda: Caiman_online(root))
processmenu.add_separator()
processmenu.add_command(label="CaImAn - Offline", command=lambda: Caiman_offline(root))


#******** REVIEW DATA MENU *************
reviewmenu = Menu(menu)
menu.add_cascade(label="Review", menu=reviewmenu)
reviewmenu.add_command(label="Review ROIs", command=lambda: Review_ROIs(root))
reviewmenu.add_separator()
reviewmenu.add_command(label="Review Spikes", command=lambda: Review_spikes(root))


#******** ANALYSIS MENU *************
analysismenu = Menu(menu)
menu.add_cascade(label="Ensemble Analysis", menu=analysismenu)
analysismenu.add_command(label="Louvain Modularity", command=lambda: Louvain_modularity(root))


mainloop()
