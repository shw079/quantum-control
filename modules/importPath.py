import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

"""!@namespace modules.importPath

!@breif This module exists to hold the import_path class, which takes user input and returns the data array for the desired path. All inputs come through the Tkinter GUI that is opened for the user. The user is given the option to (1) Draw their path with the cursor, (2) Load a path from file, or (3) Load a function from file.

"""

def import_my_module(full_name, path):
	"""!@breif Import a python module from a path. 3.4+ only. """
	from importlib import util

	spec = util.spec_from_file_location(full_name, path)
	mod = util.module_from_spec(spec)

	spec.loader.exec_module(mod)
	return mod

class import_path(tk.Tk):

	'''!@breif This class provides the viualization window for the user to draw a path. 
	The coordinates of the path are then recorded in a list for future use
	credit to: stack_exchange_40604233 for base design'''

	def __init__(self):
		'''!@breif Initialize class; create canvas for gui and set initial variables'''
		tk.Tk.__init__(self)
		self.width = 600
		self.height = 600
		self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg = "black")
		self.canvas.pack(side="top", fill="both", expand=True)
		
		#intialize variables
		self.counter = 0
		self.previous_x = self.current_x = int(self.width/2.0)
		self.previous_y = self.current_y = int(self.height/2.0)
		self.coordinate_array = np.array([])
		
		#add buttons
		self.button_help = tk.Button(self, text = "Help", command = self.instructions)
		self.button_help.pack(side="top", fill="both", expand=True)
		
		self.button_clear = tk.Button(self, text = "Select File", command = self.load_from_file)
		self.button_clear.pack(side="top", fill="both", expand=True)
		
		self.button_clear = tk.Button(self, text = "Clear", command = self.clear)
		self.button_clear.pack(side="top", fill="both", expand=True)
		
		self.button_done = tk.Button(self, text = "Done", command = self.finished)
		self.button_done.pack(side="top", fill="both", expand=True)
		
		#bind commands
		self.bind('<B1-Motion>', self.position_previous)
		self.canvas.bind('<B1-Motion>', self.draw_line)
		self.bind('<B1-Motion>',self.record_coordinates)
		
	def __del__(self):
		return
		
	def instructions(self):
		'''!@breif Display instuctions when 'instructions' button clicked'''

		messagebox.showinfo("Help",
		"To draw a path, simply click and drag in the black space provided. If you would like to load from file (either data or a function) please use the 'Select File' button. When  you are finished, click 'Done'. To erase any drawn or imported data, click 'Clear'.")

	def position_previous(self,event):
		'''Need the track the previous position for drawing lines'''
	
		self.previous_x = event.x
		self.previous_y = event.y

	def draw_line(self, event):
		'''!@breif Visualize the path as it's being drawn'''
	
		#if this is the first click, intialize near the click
		if self.counter == 0:
			self.previous_x = event.x + 1
			self.previous_y = event.y
	
		self.current_x = event.x
		self.current_y = event.y

		self.canvas.create_line(self.previous_x, self.previous_y, 
			self.current_x, self.current_y,
			fill="white")
			
		self.previous_x = event.x
		self.previous_y = event.y
		
		self.counter += 1
		
	def record_coordinates(self, event):
		'''!@breif Keep every coordinate in a list, but not repeating coordinates
		NOTE: need to subtract y from height since pixels are recorded from top'''
		
		if len(self.coordinate_array) == 0:
			self.coordinate_array = np.array([[event.x, self.height - event.y]])
		#don't record duplicates
		else:
			if event.x != self.coordinate_array[-1,0] or event.y != self.coordinate_array[-1,1]:
				self.coordinate_array = np.row_stack((self.coordinate_array, np.array([[event.x, self.height - event.y]])))
				
	def clear(self):
		'''!@breif Clear all data held in the object and start over'''
		
		#clear canvas
		self.canvas.delete("all")
		
		#re-instatiate variables
		#intialize variables
		self.counter = 0
		self.previous_x = self.current_x = int(self.width/2.0)
		self.previous_y = self.current_y = int(self.height/2.0)
		self.coordinate_array = np.array([])
		
	def load_from_file(self, filename=None):
		'''!@breif Allow user to choose file for input'''
		if filename == None:
			filename = filedialog.askopenfilename(initialdir="./", title='Please select a file')
				
		#is file an analytic function?
		#if it is, import user_function() from file and assign output to coordinate_array
		#else if data, import data
		filename_no_ext, file_ext = os.path.splitext(filename)
		if file_ext == '.py':
			user_module = import_my_module(filename_no_ext, filename)
			self.coordinate_array = user_module.user_function() #function MUST be named user_function()
		elif file_ext == '.dat':
			self.coordinate_array = pd.read_table(filename, sep=" ", header=None)
		else:
			raise ValueError("Path provided must be to a file with either the '.py' (function) or '.dat' (data) extenstion")
			
	def finished(self):
		self.destroy()
		
	def get_coordinates(self):
		'''!@breif Returns the list of coordinates as numpy array
                
                @reutrn Coorinates from the user input (2xn)  
                '''

		coords = np.array(self.coordinate_array)
		if len(coords) == 0:
			raise ValueError("Error: No coordinates present")
			return np.array([])
		#make sure starts at (0,0)
		coords = coords - coords[0]
		return coords
		
	def plot_coordinates(self):
		'''!@breif Plot coordinates held in coordinate list'''

		coords = self.get_coordinates()
		if len(coords) == 0:
			raise ValueError("Error: No coordinates present")
			return
		
		#subsample if more than 10,000 coordinates
		if len(coords) >= 10000:
			dp = int(len(coords)/1000)
			coords_p = coords[::dp]
		else:
			coords_p = coords
		color_idx = np.linspace(0, 1, len(coords_p))
		for i in range(0,len(coords_p)):
			plt.plot([coords_p[i,0]],[coords_p[i,1]],'o',color=plt.cm.cool(color_idx[i]))
		plt.show()

if __name__ == "__main__":

	#create gui object, name main window root
	root = import_path()

	#start event driven loop
	root.mainloop()
	
	#plot to check
	root.plot_coordinates()
	
	
	
	
	
	
