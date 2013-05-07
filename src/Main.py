#load remote and write to local file
import urllib2
url = 'http://aima.cs.berkeley.edu/data/iris.csv'
u = urllib2.urlopen(url)
localFile = open('iris.csv','w')
localFile.write(u.read())
localFile.close()
#CSV file parse procedure.
from numpy import genfromtxt
#read the first 4 columns
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
#read the fifth column
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)
#print
print data.shape
print target.shape
# build a collection of unique elements
print set(target) 
# pyplot visualization
from pylab import plot,show
#plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
#plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
show()