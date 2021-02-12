import matplotlib.pyplot as plt

labels = [1,0,4,5,2,1,1,0,4,2,4,4,1,2,4]

plt.hist(labels, bins=[0,1,2,3,4,5])
plt.title('Title')
plt.xlabel('Horizontal Axis')
plt.ylabel('Vertical Axis')
plt.show()