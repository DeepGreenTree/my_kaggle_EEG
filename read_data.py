"""
Cache the train and test data via NumPy save()

The data is saved as an array of the form [data, label]
id,Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10
subj1_series1_0,-31,363,211,121,211,15,717,279,35,158,543,-166,192,230,573,860,128,59,272,473,325,379,536,348,383,105,607,289,459,173,120,704
subj1_series1_1,-29,342,216,123,222,200,595,329,43,166,495,-138,201,233,554,846,185,47,269,455,307,368,529,327,369,78,613,248,409,141,83,737
subj1_series1_2,-172,278,105,93,222,511,471,280,12,177,534,-163,198,207,542,768,145,52,250,452,273,273,511,319,355,66,606,320,440,141,62,677
subj1_series1_3,-272,263,-52,99,208,511,428,261,27,180,525,-310,212,221,542,808,115,41,276,432,258,241,521,336,356,71,568,339,437,139,58,592
每个人subj1进行多个series的测试，
每个测试里，有多个step，即series1_0,，每个step持续2ms
每个step，即series1_0，包括了32个channel的数据，
每个step对于了6个event，即action，没个action有一个label
来标注出这个action是否action了。
------股票意义-------
每个step，即series1_0对应了一天。32channel，对应的就是包括
量价2个参数之内的其他精选出的30个股票指标。
方案1：每个series，对应的是一支股票
每个sub对应的是一个板块。
方案2：每个series，对应的是一支股票的一个完整波动周期
从底到顶再到底。
每个sub对应的是一只股票。
好像在实际的数据计算中，并没有区分每个series之间的区别
实际上好像全都读入到了一个文件中。
在读程序时注意这点
从性质上说，一支股票对应一个人sub，是有相似性的。
每个人的脑电图有差异，每支股票也有差异。
人的脑电图的波动，应该没有股票的波动大
这个程序提供了这么多的model，可以一个一个的试，
找出预测股票最准的。
------股票意义-------
For train, the labels are those supplied.
For test, the labels are set to zero.

One trial is defined as the progressive sequence
of actions from the first to the sixth action.
The six actions that we wish to predict are
HandStart
FirstDigitTouch
BothStartLoadPhase
LiftOff
Replace
BothReleased

In particular, the EEG signal for each trial consists of
a real value for each of the 32 channels at every
time step in the signal.
The subject’s EEG responses are sampled at 500 Hz,
and so the time steps are 2ms from each other.
For each time step, we are provided with six labels,
describing which of the six actions are active
at that time step.
"""
import os

import numpy as np
import pandas as pd

csvdir = 'data/train'#the directory or folder of training data
n_subs = 12#12 subjects，12个人
n_series = 8#8 trails，每人8次实验
n_channels = 32 #32 electric nodes，每次实验，收集32个电极的数据

data = []#creat a empty list
#把所有人的所有series的数据都放在data列表里
#data[row,column],row是每个人
#column是每个series。
#每个column里是channels的数据，32个channels
#每个column的名字
#'Fp1'	'Fp2'	'F7'	'F3'	'Fz'	'F4'	'F8'	'FC5'	'FC1'	'FC2'	'FC6'	'T7'	'C3'	'Cz'	'C4'	'T8'	'TP9'	'CP5'	'CP1'	'CP2'	'CP6'	'TP10'	'P7'	'P3'	'Pz'	'P4'	'P8'	'PO9'	'O1'	'Oz'	'O2'	'PO10'
#这些column的名字就是电极在头上的分布
#保存在数据文件在hs.eeg.names和hs.eeg.sig两个文件里

label = []
#label有6列

"""
1.HandStart
2.FirstDigitTouch
3.BothStartLoadPhase
4.LiftOff
5.Replace
6.BothReleased

有可能每个action有多个动作相关
"""
#########
# Train #
#########

# For each subject
for sub in np.arange(n_subs):
    sub_data = []#creat a empty list.
    #存放每个人的全部series的数据
    #最后把所有人的数据都放在data列表里
    sub_label = []

    # For each series
    for series in np.arange(n_series):

        # Read this data
        csv = 'subj' + str(sub + 1) + '_series' + str(series + 1) + '_data.csv'
        #注意此处是data文件，下面还有一个event文件
        #label存在event文件里
        series_data = pd.read_csv(os.path.join(csvdir, csv))
        #pd.read_csv（）函数是把csv文件里的所有数据都
        #读到series_data 的DataFrame里来。不是仅仅读一行
        #DataFrame是一个加了表头和index的矩阵
        #series_data里是放的是nX33矩阵的数据
        #包括了所有500Hz取样点的全部电极的采样数据
        #n是这个series里的采样次数，33是id列+电极数32

        # Add the data (without the ids) to our collection
        ch_names = list(series_data.columns[1:])#from the 2nd column
        #注意！！！
        # series_data.columns方法是读取dataframe的列名。
        #它不是读取series_data里从第一列开始的所有数据！！！
        #ch_names = [Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10]
        #就是32个channel的名字！！
        #idx = df.columns # get col index(name)
        #label = df.columns[0] # 1st col label
        # list()把DataFrame格式的数据(第一行的列名)换成list
        #ch_names里存放的不是这个人的series的全部数据
        #是切掉了id的第一行的列名
        #ch_names里的是1X32列表的数据，切掉了id列
        #ch_names stands for channel name
        series_data = np.array(series_data[ch_names], 'float32')
        # series_data[ch_names]就是把相应column name的数据选出
        #因为这是dataframe，colume name row name是可以从名字上区分出来的
        # series_data[ch_names]选出的数据，只有数据，没有列名了
        # 然后把这些数据转成数组，浮点的        #
        #并转换成数组。
        #数组与list的区别。list带逗号，array不带
        #[1,2,3,4]是list
        #[1 2 3 4]是array
        #看来append()操作前，需要把数据都转成数组才能append
        #注意此处不是：
        # series_data = np.array([ch_names], 'float32')
        #这样写是错误的。以为这是字符串！！！不是数据
        #在每一个for n_series的循环，ch_names 和series_data都会
        #被刷新一次。在刷新前都存在了sub_data列表里。
        #所以这个程序应该是把所有的each series都放在了一维的list里了。
        #
        sub_data.append(series_data)
        #sub_data里的数据的排列形式就是，如下：
        """
        [series_data1, series_data2, series_data3]
        """
        #这样的sub_data数组。这是某个人的所有series数据
        # 每个series_data里面是
        #32个channel的实际数据，没有ID，有ID那就是32个column了
        #把所有的each series都放在了一维的list里了
        #所有每个series_data里放的是nX32矩阵的数据
        #n是这个series里的采样次数，32是电极数
        #sub_data里的数据是nX(8X32)的矩阵数据
        #即sub_data里的数据是nX256矩阵的数据
        #append the array series_data into
        #data[num of sub_data, num of series_data]
        #这里是嵌套关联，不是二维坐标的行列关系

        # Read the corresponding events
        csv = 'subj' + str(sub + 1) + '_series' + str(series + 1) + '_events.csv'
        series_label = pd.read_csv(os.path.join(csvdir, csv))

        # Add the events (without the ids) to our collection
        ch_names = list(series_label.columns[1:])
        #就是把DataFrame转换成列表
        series_label = np.array(series_label[ch_names], 'float32')
        sub_label.append(series_label)
    # for n_series 循环结束
    data.append(sub_data)#for n_series 循环结束
    #data数组里面的数据排列形式如下：
    """
    [sub_data1, sub_data2, sub_data3, sub_data4]
    """
    # 这样的data数组。data列表里是所有12个人的全部数据
    # 每个sub_data里面就是某个人的所有series的数据，
    #每个series里就是这个series的全部
    # 32个channel的实际数据，
    #比如data[1,1],指的就是，第1行，第一列的那个数据
    #就是第二个人的的第二个series的数据。
    #列号1，代表第二个人，行号1，代表第二个series。
    #data里的数据就是nX(12X8X32)的矩阵数据
    #即data里的数据是nX3072矩阵的数据
    label.append(sub_label)

# Save
np.save('eeg_train.npy', [data, label])
#.npy文件是把数组保存在文件里的方式
#此处相当于把两个csv文件合并在了一个npy文件里
#纯数据，没有列名，和行名
########
# Test #
########

csvdir = 'data/test'
n_subs = 12
n_series = 2
n_channels = 32

data = []
label = []

# For each subject
for sub in np.arange(n_subs):
    sub_data = []
    sub_label = []

    # For each series
    for series in np.arange(9, 9 + n_series):

        # Read the data
        csv = 'subj' + str(sub + 1) + '_series' + str(series) + '_data.csv'
        series_data = pd.read_csv(os.path.join(csvdir, csv))

        # Add the data (without the ids) to our collection
        ch_names = list(series_data.columns[1:])
        series_data = np.array(series_data[ch_names], 'float32')
        sub_data.append(series_data)

        # Add placeholder labels to our collection
        series_label = np.zeros([series_data.shape[0], 6])
        sub_label.append(series_label)

    data.append(sub_data)
    label.append(sub_label)

np.save('eeg_test.npy', [data, label])
