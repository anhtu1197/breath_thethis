import numpy as geek 
  
a = geek.arange(5) 
  
# a is printed. 
print("a is:") 
print(a) 
  
# the array is saved in the file geekfile.npy  
geek.save('geekfile', a) 
  
print("the array is saved in the file geekfile.npy") 



  
# the array is loaded into b 
b = geek.load('geekfile.npy') 
  
print("b is:") 
print(b) 
  
# b is printed from geekfile.npy 
print("b is printed from geekfile.npy") 