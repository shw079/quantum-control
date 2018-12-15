''' main.py
'''

user_path = importPath()
user_path.mainloop()
data = DataStore(user_path.get_coordinates)

#calc field
solver = PathToField( data.get_path() )
solver.calc_field()
data.set_field( solver.get_field() )

#noise analysis
 
v = vis(data)
v.plot()
