import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
from os import listdir


def createDataSet():
    # 四组二维特征
    group = np.array([[1, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 四组特征的标签
    labels = ['A', 'A', 'B', 'B']
    return group, labels



#  inX - 用于分类的数据(测试集)
#  dataSet - 用于训练的数据(训练集)
#  labes - 分类标签
#  k - kNN算法参数,选择距离最小的k个点
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape[0] 读取矩阵行的长度
    # 距离计算
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile https://www.codenong.com/js0b86a2768fba/
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 距离排序
    sortedDistIndicies = distances.argsort()
    # 统计距离最近前k个的类别
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2Matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 创建返回的Numpy矩阵
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        # 删除空白行
        line = line.strip()
        listFromLine = line.split('\t')
        # 选取前3个元素（特征）存储在返回矩阵中
        returnMat[index, :] = listFromLine[0:3]
        # -1索引表示最后一列元素,位label信息存储在classLabelVector
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# datingDataMat - 特征矩阵
# datingLabels - 分类Label
def showdatas(datingDataMat, datingLabels):
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=0.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数')
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6)
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6)
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6)
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses],labels=['不喜欢', '魅力一般','极具魅力'])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses],labels=['不喜欢', '魅力一般','极具魅力'])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses],labels=['不喜欢', '魅力一般','极具魅力'])
    # 显示图片
    plt.show()


def autoNorm(dataSet):
    # 获得数据的最小值和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回dataSet的行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals


def datingClassTest():
    # 打开的文件名
    filename = "datingTestSet2.txt"
    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2Matrix(filename)
    # 取所有数据的百分之十
    hoRatio = 0.10
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果:%s\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))


def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    # 打开的文件名
    filename = "datingTestSet2.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2Matrix(filename)
    # 训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult - 1]))

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    # 样本数据的类标签列表
    hwLabels = []
 
    # 样本数据文件列表
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    # 初始化样本数据矩阵（M*1024）
    trainingMat = np.zeros((m, 1024))
    # 依次读取所有样本数据到数据矩阵
    for i in range(m):
        # 提取文件名中的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr) 
        # 将样本数据存入矩阵
        trainingMat[i, :] = img2vector(
            'trainingDigits/%s' % fileNameStr)
    # 循环读取测试数据
    testFileList = listdir('testDigits')
    # 初始化错误率
    errorCount = 0.0
    mTest = len(testFileList)
    # 循环测试每个测试数据文件
    for i in range(mTest):
        # 提取文件名中的数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # 提取数据向量
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 对数据文件进行分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # 打印 K 近邻算法分类结果和真实的分类
        print("测试样本 %d, 分类器预测: %d, 真实类别: %d" %
              (i+1, classifierResult, classNumStr))
        # 判断K 近邻算法结果是否准确
        if (classifierResult != classNumStr):
            errorCount += 1.0
    # 打印错误率
    print("\n错误分类计数: %d" % errorCount)
    print("\n错误分类比例: %f" % (errorCount/float(mTest)))

if __name__ == "__main__":
    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    testVector = img2vector('testDigits/0_13.txt')
    handwritingClassTest()    