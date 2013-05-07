#load remote and write to local file
import urllib2
url = 'http://aima.cs.berkeley.edu/data/iris.csv'
u = urllib2.urlopen(url)
localFile = open('iris.csv','w')
localFile.write(u.read())
localFile.close()
#CSV file parse procedure.
from numpy import genfromtxt,zeros
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
plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
show()
# histogram visualization
from pylab import figure,subplot,hist,xlim
xmin = min(data[:,0])
xmax = max(data[:,0])
figure()
subplot(411)#distribution of setosa class(1st,on the top)
hist(data[target=='setosa',0],color='b',alpha=.7)
xlim(xmin,xmax)
subplot(412) # distribution of the versicolor class (2nd)
hist(data[target=='versicolor',0],color='r',alpha=.7)
xlim(xmin,xmax)
subplot(413) # distribution of the virginica class (3rd)
hist(data[target=='virginica',0],color='g',alpha=.7)
xlim(xmin,xmax)
subplot(414) # global histogram (4th, on the bottom)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)
show()
# classification
t = zeros(len(target))
t[target=='setosa'] = 1
t[target=='versicolor'] = 2
t[target=='virginica'] = 3
# ready to instantiate and train classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(data, t)#training on the iris dataset
# predict
print classifier.predict(data[0])
print t[0]
# wider range of samples
from sklearn import cross_validation
train,test,t_train,t_test = cross_validation.train_test_split(data,t,test_size=0.4,random_state=0)
classifier.fit(train,t_train)#train
print classifier.score(test, t_test)#test
# fusion matrix
from sklearn.metrics import confusion_matrix
print confusion_matrix(classifier.predict(test),t_test)
# fusion matrix report
from sklearn.metrics import classification_report
print classification_report(classifier.predict(test),t_test,target_names=['setos','versicolor','virginica'])