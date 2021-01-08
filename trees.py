from math import log

#给定数据集的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)      #返回数据集的行数
    labelCounts = {}               #保存每个标签（label）出现次数的字典
    for featVec in dataSet:        #对每组特征向量进行统计
        # 为所有可能分类创建字典
        currentLabel = featVec[-1]    #提取标签信息
        if currentLabel not in labelCounts.keys():  #如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0           
        labelCounts[currentLabel] += 1                #Label计数
    shannoEnt = 0.0                            #香农熵
    for key in labelCounts:                   #计算香农熵
        # 以2为底求对数
        prob = float(labelCounts[key])/numEntries    #选择该标签(Label)的概率
        shannoEnt -= prob * log(prob,2)              #利用公式计算
    return shannoEnt                                 #返回香农熵

def creatDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'yes'],[0,1,'no'],[0,1,'no']]        #数据集
    labels = ['不放贷','放贷']                       #分类属性
    return dataSet,labels                           #返回数据集和分类属性



if __name__ == "__main__":
    myDat, dataSet = creatDataSet()
    print(calcShannonEnt(myDat))


