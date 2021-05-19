
import nexusformat.nexus as nx # at the moment just needed for debugging purposes

f = nx.nxload("ReferenceProjection.hdf5")
print(f.tree)

type_data = ""
filename = "ReferenceProjection.hdf5"
with h5py.File(filename, "r") as f:
    #f.visititems(visitor_func)

    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    
    # Get the data
    #data = list(f[a_group_key])
    #data = f.get(a_group_key)
    dimension = f.get("Dimension")
    type_dataset = f.get("Type")

    dimension_data = np.zeros(dimension.shape)
    type_data = np.zeros(type_dataset.shape)

    dimension.read_direct(dimension_data)
    type_dataset.read_direct(type_data)

    """
      for item in data: 
         print(item)
      
      print(len(data))
      print(data)
      print(data[0])
      print(type(data[0]))
    """
    np.savetxt('dimension_hdf5.txt', dimension_data, delimiter=',') 
    np.savetxt('type_hdf5.txt', type_data, delimiter=',') 


f = nx.nxload("".join([args.output_file_name, ".hdf5"]))


print(f.tree)


"""
Stuff from AI-CIT
"""
self.num_samples_in_x = 0
while(self.stride*self.num_samples_in_x+self.num_pixel <= self.x): 
    self.num_samples_in_x+=1 
self.num_samples_in_z = 0
while(self.stride*self.num_samples_in_z+self.num_pixel <= self.z): 
      self.num_samples_in_z+=1 
