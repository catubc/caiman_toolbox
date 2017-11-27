from Tkinter import *
import os
import bokeh 

from menu_utils import  *
    
root = Tk()
menu = Menu(root)
root.config(menu=menu)
root.minsize(width=1200, height=500)
root.title("CaImAn GUI and Analysis Toolbox")
root.geometry('250x150+200+100')


#************************************************************
#************************ FILE MENU *************************
#************************************************************
filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="New", command=NewFile)
filemenu.add_command(label="Open...", command=OpenFile)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

#************************************************************
#*********************** PREPROCESS MENU ********************
#************************************************************
preprocessmenu = Menu(menu)
menu.add_cascade(label="Pre-Process", menu=preprocessmenu)
preprocessmenu.add_command(label="Image Registration", command=lambda: Image_registration(root))
preprocessmenu.add_command(label="Convert .tif -> .npy", command=lambda: Tif_convert(root))

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
reviewmenu.add_command(label="View Rasters", command=lambda: View_rasters(root))


#******** ANALYSIS MENU *************
analysismenu = Menu(menu)
menu.add_cascade(label="Analysis", menu=analysismenu)
analysismenu.add_command(label="Ensemble Detection", command=Ensemble_detection)



#******** EXPORT DATA MENU ************


mainloop()
