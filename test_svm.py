from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plot

clf = svm.SVC(gamma=0.001, C=100.)
# # iris
# iris = datasets.load_iris()
# clf.fit(iris.data[:-1], iris.target[:-1])
# result=clf.predict(iris.data[-5:])
# print "predict: ", result
# print "actural: \n", iris.data[-5:], "\n", iris.target[-5:]

# digits
digits = datasets.load_digits()
clf.fit(digits.data[:-1], digits.target[:-1])
# result1=clf.predict(digits.data[-2])
# plot.figure(1, figsize=(3, 3))
# plot.imshow(digits.images[-2], cmap=plot.cm.gray_r, interpolation='nearest')
# plot.show()
# print "digits.images[-2]: ", digits.images[-2]
# print "predict: ", result1
# print "actural: \ndigits.data[-2]:\n", digits.data[-2], "\ndigits.target[-2]:\n", digits.target[-2]
my_sample = [0, 15, 7, 9, 16, 7, 15, 0,
             0, 8, 10, 12, 7, 6, 14, 0,
             0, 9, 1, 1, 2, 9, 7, 0,
             0, 10, 11, 16, 6, 13, 13, 0,
             0,	11, 13, 12, 9, 12, 5, 0,
             0, 1, 1, 1, 2,	9, 10, 0,
             0,	10, 11, 12, 13, 10, 11,	0,
             0, 9, 7, 9, 11, 15, 13, 0]

my_sample_pic = [[0, 15, 7, 9, 16, 7, 15, 0],
             [0, 8, 10, 12, 7, 6, 14, 0],
             [0, 9, 1, 1, 2, 9, 7, 0],
             [0, 10, 11, 16, 6, 13, 13, 0],
             [0, 11, 13, 12, 9, 12, 5, 0],
             [0, 1, 1, 1, 2, 9, 10, 0],
             [0, 10, 11, 12, 13, 10, 11, 0],
             [0, 9, 7, 9, 11, 15, 13, 0]]

result2=clf.predict(my_sample)
plot.figure(1, figsize=(3, 3))
plot.imshow(my_sample_pic, cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()

print "\npredict: ", result2
print "actural: \nmy_sample:\n", my_sample, "\nmy_sample ans is '9'"