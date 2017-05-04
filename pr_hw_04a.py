#-*- coding: utf-8 -*-
import csv
import numpy
import random
import matplotlib.pyplot as plt
def read_file():
    file=open("iris.data.txt","r")
    original_data_set=[]
    for i in csv.reader(file):
        for j in range(5):
            original_data_set.append(i[j])
    return original_data_set
def split_data(original_data_set):#,percentage,class_number):
    #original_data_set=[1,2,3,4,5,6,7,8,9,10]
    #original_data_set=numpy.reshape(original_data_set,[len(original_data_set)/5,5])
    training_set=[]
    test_set=[]
    #print(numpy.delete(original_data_set,0,0))
    for i in range(len(original_data_set)):
        if i%10<5:
            training_set.append(original_data_set[i])
        else:
            test_set.append(original_data_set[i])
    return training_set,test_set
def one_perceptron(training_set,test_set):
    learning_rate=numpy.array([0.1,0.1,0.1,0.1,0.1],float)
    weight=numpy.array([0.5,random.randrange(-1,1),random.randrange(-1,1),random.randrange(-1,1),random.randrange(-1,1)])
    training_set=numpy.reshape(training_set,[len(training_set)/5,5])
    test_set=numpy.reshape(test_set,[len(test_set)/5,5])
    #print weight
    for j in range(20):
        #print "========================"
        for i in training_set:
            x=[-1,float(i[0])/10,float(i[1])/10,float(i[2])/10,float(i[3])/10]
            #print numpy.dot(weight.T,x)
            if numpy.dot(weight.T,x)>=0:
                print(i[4])
                if i[4]!="Iris-setosa":
                    weight=numpy.add(weight,-learning_rate*x)
                    #print weight
                    print(j," X")
                else:
                    print (j," O")
            else:
                print(i[4])
                if i[4]!="Iris-versicolor":
                    weight=numpy.add(weight,learning_rate*x)
                    print (weight)
                    print (j,"X")
                else:
                    print (j,"O")
    for i in test_set:
        if numpy.dot(weight.T,x)>=0:
            if i[4]!="Iris-setosa":
                plt.plot(i[0],i[2],'ro')
                #weight=weight-learning_rate*x
                    #print weight
                #print j," X"
            else:
                plt.plot(i[0],i[2],'bo')
                #print j," O"
        else:
            if i[4]!="Iris-versicolor":
                plt.plot(i[0],i[2],'ro')
                #print weight
                #print j,"X"
            else:
                plt.plot(i[0],i[2],'go')
                #print j,"O"
    print(weight)
    plt.show()
if __name__ == '__main__':
    #percentage=50#每一類取50%出來訓練
    #class_number=3#資料內有幾類
    original_data_set=read_file()
    training_set,test_set=split_data(original_data_set)#,percentage,class_number)
    one_perceptron(training_set,test_set)