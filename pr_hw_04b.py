#-*- coding: utf-8 -*-
import csv
import random
import math
import numpy
import matplotlib.pyplot as plt
learning_rate=0.1
def read_file():#讀檔
    file=open("iris2.data","r")
    original_data_set=[]
    for i in csv.reader(file):#解析CSV檔案
        for j in range(5):
            original_data_set.append(i[j])
    return original_data_set
def split_data(original_data_set):#分切資料
    training_set=[]
    test_set=[]
    for i in range(len(original_data_set)):#奇數資料當訓練 偶數當測試
        if i%10<5:
            training_set.append(original_data_set[i])
        else:
            test_set.append(original_data_set[i])
    return training_set,test_set#回傳
def sigmoid(y): #activation function
    output=1.0/(1.0+math.exp(-(y)))
    return output
#input?HIDDEN LAYER?WEIGHT
def update_weight(W1,W2,Theta,i,Hidden_Layer_Output,ans,output):#更新WEIGHT 算出該變化的量後回傳
    DW1=[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    DW2=[[0,0],[0,0],[0,0]]
    DT=[0,0,0,0,0]
    local_gradient=[0,0,0,0,0]
    local_gradient[3]=output[0]*(1-output[0])*(ans[0]-output[0])#算LOCAL_GRADIENT[3]..對應的是OUTPUT LAYER
    local_gradient[4]=output[1]*(1-output[1])*(ans[1]-output[1])#算LOCAL_GRADIENT[4]..對應的是OUTPUT LAYER
    DW2[0][0]=learning_rate*Hidden_Layer_Output[0]*local_gradient[3]#算HIDDEN[0]->OUTPUT[0] WEIGHT 該變化的值
    DW2[0][1]=learning_rate*Hidden_Layer_Output[0]*local_gradient[4]#算HIDDEN[0]->OUTPUT[0] WEIGHT 該變化的值
    DW2[1][0]=learning_rate*Hidden_Layer_Output[1]*local_gradient[3]#算HIDDEN[1]->OUTPUT[1] WEIGHT 該變化的值
    DW2[1][1]=learning_rate*Hidden_Layer_Output[1]*local_gradient[4]#算HIDDEN[1]->OUTPUT[1] WEIGHT 該變化的值
    DW2[2][0]=learning_rate*Hidden_Layer_Output[2]*local_gradient[3]#算HIDDEN[2]->OUTPUT[2] WEIGHT 該變化的值
    DW2[2][1]=learning_rate*Hidden_Layer_Output[2]*local_gradient[4]#算HIDDEN[2]->OUTPUT[2] WEIGHT 該變化的值
    DT[3]=learning_rate*(-1)*local_gradient[3]#算THETA..對應的是OUTPUT LAYER
    DT[4]=learning_rate*(-1)*local_gradient[4]#算THETA..對應的是OUTPUT LAYER
    #
    local_gradient[0]=Hidden_Layer_Output[0]*(1-Hidden_Layer_Output[0])*(local_gradient[3]*W2[0][0]+local_gradient[4]*W2[0][1])#算LOCAL_GRADIENT[0]..對應的是Hidden_Layer
    local_gradient[1]=Hidden_Layer_Output[1]*(1-Hidden_Layer_Output[1])*(local_gradient[3]*W2[1][0]+local_gradient[4]*W2[1][1])#算LOCAL_GRADIENT[1]..對應的是Hidden_Layer
    local_gradient[2]=Hidden_Layer_Output[2]*(1-Hidden_Layer_Output[2])*(local_gradient[3]*W2[2][0]+local_gradient[4]*W2[2][1])#算LOCAL_GRADIENT[2]..對應的是Hidden_Layer
    DW1[0][0]=learning_rate*float(i[0])*local_gradient[0]#算INPUT[0]->HIDDEN[0] WEIGHT 該變化的值
    DW1[0][1]=learning_rate*float(i[0])*local_gradient[1]#              :
    DW1[0][2]=learning_rate*float(i[0])*local_gradient[2]#              :
    DW1[1][0]=learning_rate*float(i[1])*local_gradient[0]#              :
    DW1[1][1]=learning_rate*float(i[1])*local_gradient[1]#              :
    DW1[1][2]=learning_rate*float(i[1])*local_gradient[2]#              :
    DW1[2][0]=learning_rate*float(i[2])*local_gradient[0]#              :
    DW1[2][1]=learning_rate*float(i[2])*local_gradient[1]#              :
    DW1[2][2]=learning_rate*float(i[2])*local_gradient[2]#              :
    DW1[3][0]=learning_rate*float(i[3])*local_gradient[0]#              :
    DW1[3][1]=learning_rate*float(i[3])*local_gradient[1]#              :
    DW1[3][2]=learning_rate*float(i[3])*local_gradient[2]#算HIDDEN[3]->OUTPUT[2] WEIGHT 該變化的值
    DT[0]=learning_rate*(-1)*local_gradient[0]#算THETA..對應的是Hidden_Layer
    DT[1]=learning_rate*(-1)*local_gradient[1]#算THETA..對應的是Hidden_Layer
    DT[2]=learning_rate*(-1)*local_gradient[2]#算THETA..對應的是Hidden_Layer
    return DW1,DW2,DT#回傳
def training(W1,W2,Theta,training_set, round):
    Hidden_Layer_Output=[0,0,0]
    output=[0,0]
    for k in range(round):#500回合
        for i in training_set:#將訓練資料每筆讀入
            for j in range(3):
                Hidden_Layer_Output[j]=sigmoid(float(i[0])*W1[0][j]+float(i[1])*W1[1][j]+float(i[2])*W1[2][j]+float(i[3])*W1[3][j]-Theta[j])
            for j in range(2):
                output[j]=sigmoid(Hidden_Layer_Output[0]*W2[0][j]+Hidden_Layer_Output[1]*W2[1][j]+Hidden_Layer_Output[2]*W2[2][j]-Theta[j+3])
            if i[4]=="Iris-setosa":
                ans=[1,0]#希望的答案
            elif i[4]=="Iris-versicolor":
                ans=[0,1]#希望的答案
            else:
                ans=[0,0]#希望的答案
            DW1,DW2,DT=update_weight(W1,W2,Theta,i,Hidden_Layer_Output,ans,output)#算需變化的WEIGHT量
            W1=numpy.add(W1,DW1)#更新WEIGHT
            W2=numpy.add(W2,DW2)#更新WEIGHT
            Theta=numpy.add(Theta,DT)#更新WEIGHT
    return W1,W2,Theta
def testing(final_W1,final_W2,final_Theta,test_set):#測試
    W1=final_W1
    W2=final_W2
    Theta=final_Theta
    Hidden_Layer_Output=[0,0,0]
    output=[0,0]
    COUNT=0
    for i in test_set:
        for j in range(3):
            Hidden_Layer_Output[j]=sigmoid(float(i[0])*W1[0][j]+float(i[1])*W1[1][j]+float(i[2])*W1[2][j]+float(i[3])*W1[3][j]-Theta[j])
        for j in range(2):
            output[j]=sigmoid(Hidden_Layer_Output[0]*W2[0][j]+Hidden_Layer_Output[1]*W2[1][j]+Hidden_Layer_Output[2]*W2[2][j]-Theta[j+3])
            if output[j]<=0.5:
                output[j]=0
            else:
                output[j]=1
        if output[0]==1 and output[1]==0:
            if i[4]=="Iris-setosa":
                #plt.plot(i[0],i[2],'go')
                COUNT=COUNT+1
            #else:
                #plt.plot(i[0],i[2],'ro')
        elif output[0]==0 and output[1]==1:
            if i[4]=="Iris-versicolor":
                #plt.plot(i[0],i[2],'yo')
                COUNT=COUNT+1
            #else:
                #plt.plot(i[0],i[2],'ro')
        elif output[0]==0 and output[1]==0:
            if i[4]=="Iris-virginica":
                #plt.plot(i[0],i[2],'bo')
                COUNT=COUNT+1
            #else:
                #plt.plot(i[0],i[2],'ro')
    #plt.show()#秀出繪製的圖檔
    print(COUNT/float(75))#正確數除以測試資料數,若為TEST 是75 全部150
    return COUNT/float(75)
if __name__ == '__main__':
    round = 1000
    #給予W1(INPUT->HIDDEN)之間所有WEIGHT初始值讓他介於-0.6~0.6之間的小數亂數
    W1=[[random.uniform(-0.6,0.6),random.uniform(-0.6,0.6),random.uniform(-0.6,0.6)],
        [random.uniform(-0.6,0.6),random.uniform(-0.6,0.6),random.uniform(-0.6,0.6)],
        [random.uniform(-0.6,0.6),random.uniform(-0.6,0.6),random.uniform(-0.6,0.6)],
        [random.uniform(-0.6,0.6),random.uniform(-0.6,0.6),random.uniform(-0.6,0.6)]]
    #W2(HIDDEN LAYER->OUTPUT)之間所有WEIGHT初始值讓他介於-0.6~0.6之間的小數亂數
    W2=[[random.uniform(-0.6,0.6),random.uniform(-0.6,0.6)],
        [random.uniform(-0.6,0.6),random.uniform(-0.6,0.6)],
        [random.uniform(-0.6,0.6),random.uniform(-0.6,0.6)],]
    #所有Theta初始值讓他介於-0.6~0.6之間的小數亂數
    Theta=[random.uniform(-0.6,0.6),random.uniform(-0.6,0.6),random.uniform(-0.6,0.6),random.uniform(-0.6,0.6),random.uniform(-0.6,0.6)]
    original_data_set=read_file()#讀檔案
    training_set,test_set=split_data(original_data_set)#切DATA
    training_set=numpy.reshape(training_set,[len(training_set)/5,5])#將資料轉成陣列
    test_set=numpy.reshape(test_set,[len(test_set)/5,5])#將資料轉成陣列
    original_data_set=numpy.reshape(original_data_set,[len(original_data_set)/5,5])#將資料轉成陣列
    plot_x = []
    plot_y = []
    for i in range(round/10):
        final_W1,final_W2,final_Theta=training(W1,W2,Theta,training_set, (i+1)*10)#訓練回傳final_W1,final_W2,final_Theta
    #print(final_W1)#訓練後的W1
    #print(final_W2)#訓練後的W2
    #print(Theta)#訓練後的THETA
        accuracy = testing(final_W1,final_W2,final_Theta,test_set)#測試
        plot_x.append((i+1)*10)
        plot_y.append(1-accuracy)
    plt.plot(plot_x, plot_y)
    plt.xlabel("learing time")
    plt.ylabel("error")
    plt.show()