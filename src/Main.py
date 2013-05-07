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
# cross validation
from sklearn.cross_validation import cross_val_score
# cross validation with 6 iterations
scores = cross_val_score(classifier,data,t,cv=6)
print scores
# mean
from numpy import mean
print mean(scores)
# k-mean clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,init='random')#initialization
kmeans.fit(data)# actual execution
c = kmeans.predict(data)
# completeness and homogeneity score
from sklearn.metrics import completeness_score,homogeneity_score
print completeness_score(t,c)
print homogeneity_score(t,c)
# visualization it
figure()
subplot(211) # top figure with the real classes
plot(data[t==1,0],data[t==1,2],'bo')
plot(data[t==2,0],data[t==2,2],'ro')
plot(data[t==3,0],data[t==3,2],'go')
subplot(212) # bottom figure with classes assigned automatically
plot(data[t==1,0],data[t==1,2],'bo',alpha=.7)
plot(data[t==2,0],data[t==2,2],'go',alpha=.7)
plot(data[t==0,0],data[t==0,2],'mo',alpha=.7)
show()
# Regression
from numpy.random import rand
x = rand(40,1) # explanatory variable
y = x*x*x + rand(40,1)/5 # depentend variable
# best-fit line regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x, y)
# plot it
from numpy import linspace,matrix
xx = linspace(0,1,40)
plot(x,y,'o',xx,linreg.predict(matrix(xx).T),'--r')
show()
# quantify the mean squared error
from sklearn.metrics import mean_squared_error
print mean_squared_error(linreg.predict(x),y)
# correlation
from numpy import corrcoef
corr = corrcoef(data.T) # .T gives the transpose
print corr
