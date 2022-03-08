#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 00:14:05 2021

@author: khushalmodi
"""
import constants
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk

class MCQ:
    def __init__(self, images_folder, images, query_folder):
        self.images_folder = images_folder
        self.images = images
        self.query_folder = query_folder
        self.rel_images = []
        self.irrel_images = []
        return
    
    def give_options(self):
        root = Tk()
        root.title("Feedback System")
        root.geometry("500x400")
   
        label1 = Label(root, text="Check the left box for marking image as relevant and right to mark it irrelevant")
        label1.pack()

        main_frame = Frame(root)
        main_frame.pack(fill=BOTH, expand=1)
        
        my_canvas = Canvas(main_frame)
        my_canvas.pack(side=LEFT, fill=BOTH, expand=0.5)
        
        my_canvas1 = Canvas(main_frame)
        my_canvas1.pack(side=RIGHT, fill=BOTH, expand=0.5)
        
        my_scroll = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
        my_scroll.pack(side=LEFT, fill=Y)
        
        my_scroll1 = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas1.yview)
        my_scroll1.pack(side=RIGHT, fill=Y)
        
        my_canvas.configure(yscrollcommand=my_scroll.set)
        my_canvas1.configure(yscrollcommand=my_scroll1.set)
        
        def _on_mousewheel(event):
            my_canvas.yview_scroll(-1*(event.delta), "units")
            
        def _on_mousewheel1(event):
            my_canvas1.yview_scroll(-1*(event.delta), "units")
        
        my_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))
        
        my_canvas1.bind_all("<MouseWheel>", _on_mousewheel1)
        my_canvas1.bind('<Configure>', lambda e: my_canvas1.configure(scrollregion=my_canvas1.bbox("all")))
        
        second_frame = Frame(my_canvas)
        my_canvas.create_window((0,0), window=second_frame, anchor="nw")
        
        second_frame1 = Frame(my_canvas1)
        my_canvas1.create_window((0,0), window=second_frame1, anchor="nw")
        
        img = []
        irrel_img = []
        rel = []
        irrel = []
        counter = 0;
        rel_cb = []
        irrel_cb = []
        rel_var = []
        irrel_var = []
        
        for image in self.images:
            if ( os.path.exists(os.path.join(self.images_folder, image)) ):
                img.append(Image.open(os.path.join(self.images_folder, image)))
            else:
                img.append(Image.open(os.path.join(self.query_folder, image)))
            
            rel.append(ImageTk.PhotoImage(img[counter]))
            
            rel_var.append(IntVar())
            
            rel_cb.append(Checkbutton(second_frame, text=image, image=rel[counter], compound='top', justify='center', variable=rel_var[counter]).pack())
            counter+=1
        
        counter = 0
        for image in self.images:
            if ( os.path.exists(os.path.join(self.images_folder, image)) ):
                irrel_img.append(Image.open(os.path.join(self.images_folder, image)))
            else:
                irrel_img.append(Image.open(os.path.join(self.query_folder, image)))
            
            irrel.append(ImageTk.PhotoImage((irrel_img[counter])))
            
            irrel_var.append(IntVar())
            
            irrel_cb.append(Checkbutton(second_frame1, text=image, image=irrel[counter], compound='top', justify='center', variable=irrel_var[counter]).pack())
            counter+=1
        
        
        root.mainloop()
        
        for i in range(0, counter):
            if ( rel_var[i].get() == 1 ):
                self.rel_images.append(self.images[i])
            elif ( irrel_var[i].get() == 1 ):
                self.irrel_images.append(self.images[i])
        
        return (self.rel_images, self.irrel_images)
            
