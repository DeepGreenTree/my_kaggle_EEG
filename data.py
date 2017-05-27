import numpy as np
import os
import skimage.io as io
import skimage.transform as tf
import sys
import numpy.random as nr
import Queue
import threading
import scipy.weave
from scipy.ndimage.interpolation import zoom

########################################################################
neg_pool = None
hard_ratio = None
easy_mode = None
bootstrap_idcs = None
offset = 0

c_code = """npy_intp dims[2]={out_length, 32};
PyObject *out_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
double *data = (double*)((PyArrayObject*)out_array)->data;
double resize_ratio = (in_length - 1.) / (out_length - 1);
double cor_loc, diff;
int x;
for(int j = 0; j < out_length - 1; j++){
    cor_loc = j * resize_ratio;
    x = (int)cor_loc;
    diff = cor_loc - x;
    for(int i = 0; i < 32; i++) {
        data[j * 32 + i] = diff * input_data[(x + 1) * 32 + i] + (1 - diff) * input_data[x * 32 + i];
    }
}
for(int i = 0; i < 32; i++) {
    data[(out_length - 1) * 32 + i] = input_data[(in_length - 1) * 32 + i];
}

return_val = out_array;
Py_XDECREF(out_array);
"""

def inline_zoom(input_data, out_length):
    """
    inline_zoom: use C inline code to resize the input data
    parameters:
        input_data: the data to be resize, shape should be [length x 32]
        out_length: the target length to be resized
    """
    assert(out_length >= 1)
    in_length = input_data.shape[0]
    input_data = np.require(input_data, requirements='C')
    output_array = scipy.weave.inline(c_code,
        ["input_data", "in_length", "out_length"],
    )

    return output_array

def bootstrap(y_true, probs):
    """
    bootstrap: bootstrap the training data, mainly used to increase volume of hard negative examples for training
    parameters:
        y_true: ground truth of training data
        probs: predictions of the model
    """
    global  neg_pool

    neg_pool_size = len(neg_pool)
    hard_size = int(neg_pool_size * hard_ratio)
    easy_size = neg_pool_size - hard_size
    neg_idcs = np.arange(len(y_true))[y_true.sum(axis = 1) == 0]
    sort_idcs = np.argsort(probs[neg_idcs, :].max(axis = 1))
    hard_idcs = bootstrap_idcs[neg_idcs[sort_idcs[-hard_size:]]]

    if easy_mode == 'random':
        easy_idcs = bootstrap_idcs[neg_idcs[nr.randint(0, len(neg_idcs), (easy_size,))]]
    elif easy_mode == 'easy':
        easy_idcs = neg_pool[:easy_size]
    elif easy_mode == 'all':
        easy_idcs = neg_pool[nr.randint(0, len(neg_pool), (easy_size,))]
    else:
        sys.exit('Wrong easy mode')

    neg_pool = np.concatenate([easy_idcs, hard_idcs])

pos_cells_train = None
neg_cells_train = None
pos_cells_valid = None
neg_cells_valid = None

def init_sample_cells(labels, events, train_series, valid_series):
    """
    init_sample_cells: get all the positive and negative indices for train series and valid series
    positive: 在time stamp中真实发生动作了的，value=1
    negative：在time stamp中没有发生动作的，value=0
    得到这些time stamp，cell， 的索引号。
    parameters:
        labels: ground truth of all the data for train and valid
        events: list of target events for training
        train_series: list of series for training
        valid_series: list of series for validation
    """
    global pos_cells_train
    global neg_cells_train
    global pos_cells_valid
    global neg_cells_valid

    pos_cells_train, neg_cells_train = sample_cells_gen(
        labels[:, train_series], events)
    pos_cells_valid, neg_cells_valid = sample_cells_gen(
        labels[:, valid_series], events)
########################################################################

def load(data_path):
    """
    load: load the prepared data
    parameters:
        data_path: path of the prepared data
    """
    data, labels = np.load(data_path)
    return data, labels

def sample_cells_gen(labels, events):
    """
    sample_cells_gen: generate all the indices of positive
    and negative samples
    parameters:
        labels: ground truth of samples
        events: list of target events for training
    """
    num_subs = labels.shape[0]#labels矩阵的行数，12个人
    num_series = labels.shape[1]#labels矩阵的列数，8个series
    pos_cells = np.zeros((0, 3), 'int32')
    #[0 0 0]
    #(0, 3)注意，Python是从0开始，但不包括3，即0,1,2,三列
    #不是1,2,3
    neg_cells = np.zeros((0, 3), 'int32')
    for sub in np.arange(num_subs):
        for series in np.arange(num_series):
            l = labels[sub, series]
            #注意此处对append出来的列表的理解
            #append（）出来的列表不是一个二维或是三维的概念
            #它是一个嵌套的概念！！！
            #最后append()的是最外的循环，最先append()的是最内的循环
            #labels[sub, series]是坐标表示，不是对列表的重新整合
            #labels[sub, series]指的是第sub个位置上的
            # 第series个位置上的那个数据
            #也就是那个ch_names，它是一个nX32矩阵的数据
            #n就是这个series里的time stamp,即cell.
            #平均n约等于20万个

            pos_idcs = np.arange(len(l))[l[:, events].sum(axis = 1) > 0]
            #numpy.arange([start, ]stop, [step, ])
            #l = labels[sub, series]
            # len(l)是求一个二维数组的长度，实质就是指array的行数
            #不是指二维数组的列数，这里就是time stamp 的数量
            #不是l的列数。len(l)=213402；
            #l就是当前的某个nX32的ch_names矩阵
            #np.arange(len(l))就是逐一的读取每一行的数据。
            #events = [0, 1, 2, 3, 4, 5]
            #l[:, events]在挑选出的time stamp的某一行的数据
            # 的[0, 1, 2, 3, 4, 5]那些列，
            # sum(axis = 1)就是，axis = 1就是在X方向求和，即把每行中的每个数相加
            #axis = 0就是在Y方向求和，即把每列中的每个数相加
            """
            a = [[ 0  1  2]
                [ 3  4  5]
                [ 6  7  8]
                [ 9 10 11]] 
             >>> a.sum(axis=0)
            array([18, 22, 26])
               >>> a.sum(axis=1)
            array([ 3, 12, 21, 30])
                
            """
            #[l[:, events].sum(axis = 1) > 0]就是把
            #这一行的数都加起来，如果大于零，则说明
            #这个time stamp里有events。
            # pos_idcs 就是在events那一列有真实动作的集合，值为1的列
            #neg_idcs就是在events那一列没有真实动作的集合，值为0的列
            neg_idcs = np.asarray(
                list(set(np.arange(len(l))).difference(set(pos_idcs))),
                'int32')
            pos = np.zeros((len(pos_idcs), 3), 'int32')
            #len(pos_idcs)就是pos_idcs的行数
            pos[:, 0] = sub
            pos[:, 1] = series
            pos[:, 2] = pos_idcs
            pos_cells = np.concatenate((pos_cells, pos), axis = 0)
            #axis = 0，在行的后面进行连接。不是增加新的列
            neg = np.zeros((len(neg_idcs), 3), 'int32')
            neg[:, 0] = sub
            neg[:, 1] = series
            neg[:, 2] = neg_idcs
            neg_cells = np.concatenate((neg_cells, neg), axis = 0)
    return pos_cells, neg_cells

def random_chunk_gen_fun(data, labels, events, params):
    """
    random_chunk_gen_fun: function to generate chunks randomly
    parameters:
        data: EEG data from which to generate the chunks
        labels: labels corresponding to the input data
        events: list of target events
        params: parameter dictionary for generation
    """
    global neg_pool
    global hard_ratio
    global easy_mode
    """
    'channels': 32,
                     'length': 3584,
                     'preprocess': 'per_sample_mean',
                     'chunk_size': 4096,
                     'num_chunks': 400,
                     'pos_ratio': 0.35,
                     'bootstrap': True,
                     'neg_pool_size': 81920,
                     'hard_ratio': 1,
                     'easy_mode': 'all',
    """

    channels = params['channels']
    length = params['length']
    num_events = len(events)
    num_chunks = params['num_chunks']
    chunk_size = params['chunk_size']
    pos_ratio = params['pos_ratio']
    pos_size = int(float(chunk_size) * pos_ratio)
    neg_size = chunk_size - pos_size

    if params['section'] == 'train':
        pos_cells, neg_cells = pos_cells_train, neg_cells_train
    elif params['section'] == 'valid':
        pos_cells, neg_cells = pos_cells_valid, neg_cells_valid
    else:
        sys.exit('Wrong section')
    sample_cells = np.concatenate((pos_cells, neg_cells), axis = 0)
    #在行的方向上连接

    ####################################
    if 'bootstrap' in params and params['bootstrap'] == True:
        if hard_ratio is None:
            hard_ratio = params['hard_ratio']
        if easy_mode is None:
            easy_mode = params['easy_mode']
        if neg_pool is None:
            neg_pool = nr.randint(0, len(neg_cells), (params['neg_pool_size'],))\
                       + len(pos_cells)
        if easy_mode is 'all':
            easy_size = len(neg_pool) - int(len(neg_pool) * hard_ratio)
            neg_pool[:easy_size] = nr.randint(0, len(neg_cells), (easy_size,))\
                                   + len(pos_cells)
    ####################################

    for i in np.arange(num_chunks):
        x_chunk = np.zeros((chunk_size, channels, 1, length), 'float32')
        #B = zeros(d1,d2,d3...) or B = zeros([d1 d2 d3...]) 返回一个d1-by-d2-by-d3-by-... .的零元素数组。
        #x_chunk = np.zeros((chunk_size, channels, 1, length))
        #返回一个chunk_size-by-channels-by-1-by-length的零元素数组。
        #x_chunk 数组的规模是：4096X32X1X3584
        y_chunk = np.zeros((chunk_size, num_events), 'int32')

        pos_chunk_idcs = nr.randint(0, len(pos_cells), (pos_size,))
        #########################################
        if 'bootstrap' in params and params['bootstrap'] == True:
            neg_chunk_idcs = np.copy(neg_pool[nr.randint(0, len(neg_pool), (neg_size,))])
        else:
            neg_chunk_idcs = nr.randint(0, len(neg_cells), (neg_size,))\
                             + len(pos_cells)
        #########################################

        chunk_idcs = np.concatenate((pos_chunk_idcs, neg_chunk_idcs))
        nr.shuffle(chunk_idcs)

        for j, idx in enumerate(chunk_idcs):
            sub = sample_cells[idx, 0]
            series = sample_cells[idx, 1]
            t = sample_cells[idx, 2]
            if 'resize' in params:
                s_len = params['resize'][0] + (params['resize'][1] - params['resize'][0]) * nr.uniform()
                s_len = int(s_len * float(length))
            else:
                s_len = length
            # Copy the history data into sample. The stop critirion for copy is smaller than t+1
            sample = np.copy(data[sub, series][max((0, t - s_len + 1)):t + 1, :])
            if params['preprocess'] == 'per_sample_mean':
                sample -= np.mean(sample, axis = 0).reshape((1, channels))
            if params['preprocess'] == 'per_sample_mean_variance':
                sample -= np.mean(sample, axis = 0).reshape((1, channels))
                sample /= (np.std(sample, axis = 0).reshape((1, channels)) + 1e-10)
            if params['preprocess'] == 'mean':
                sample -= params['mean'].reshape((1, channels))
            if params['preprocess'] == 'mean_variance':
                sample -= params['mean'].reshape((1, channels))
                sample /= params['std'].reshape((1, channels))
            if sample.shape[0] < s_len:
                sample = np.concatenate([np.zeros((s_len - sample.shape[0], channels),
                                                 'float32'),
                                         sample], axis = 0)
            if 'resize' in params:
                factor = float(length) / s_len
                #sample = zoom(sample, [factor, 1], order = 1)
                sample = inline_zoom(sample, length)

            x_chunk[j] = np.asarray(sample, 'float32').T.reshape((channels, 1, length))
            y_chunk[j, :] = labels[sub, series][t, events]

        yield x_chunk, y_chunk, chunk_size

def fixed_chunk_gen_fun(data, labels, events, params):
    """
    fixed_chunk_gen_fun: function to generate chunks with fixed order
    parameters:
        data: EEG data from which to generate the chunks
        labels: labels corresponding to the input data
        events: list of target events
        params: parameter dictionary for generation
    """
    global bootstrap_idcs
    global offset

    channels = params['channels']
    length = params['length']
    num_events = len(events)
    chunk_size = params['chunk_size']

    #########################################################################
    pos_interval = 1
    if 'pos_interval' in params:
        pos_interval = params['pos_interval']
    neg_interval = 1
    if 'neg_interval' in params:
        neg_interval = params['neg_interval']

    if 'offset' in params:
        offset = params['offset']
    else:
        offset = nr.randint(0, max(pos_interval, neg_interval))

    if params['section'] == 'valid':
        pos_cells = pos_cells_valid[offset::pos_interval, :]
        neg_cells = neg_cells_valid[offset::neg_interval, :]
    elif params['section'] == 'bootstrap':
        pos_cells = pos_cells_train[offset::pos_interval, :]
        neg_cells = neg_cells_train[offset::neg_interval, :]
        bootstrap_idcs = np.concatenate([np.arange(len(pos_cells_train))[offset::pos_interval],
                                         np.arange(len(neg_cells_train))[offset::neg_interval]\
                                         + len(pos_cells_train)])
    ##########################################################################
    sample_cells = np.concatenate((pos_cells, neg_cells), axis = 0)
    num_cells = len(sample_cells)

    num_chunks = int(np.ceil(float(num_cells) / chunk_size))
    last_chunk_length = np.mod(num_cells, chunk_size)
    if last_chunk_length == 0:
        last_chunk_length = chunk_size
    idcs_gen = lambda n: np.arange(n * chunk_size, (n + 1) * chunk_size)

    for i in np.arange(num_chunks):
        idcs = idcs_gen(i)
        x_chunk = np.zeros((chunk_size, channels, 1, length), 'float32')
        y_chunk = np.zeros((chunk_size, num_events), 'int32')
        if i != num_chunks - 1:
            chunk_length = chunk_size
        else:
            chunk_length = last_chunk_length
            idcs[chunk_length:] = 0

        for j, idx in enumerate(idcs):
            sub = sample_cells[idx, 0]
            series = sample_cells[idx, 1]
            t = sample_cells[idx, 2]

            # Copy the history data into sample. The stop critirion for copy is smaller than t+1
            sample = np.copy(data[sub, series][max((0, t - length + 1)):t + 1, :])
            if params['preprocess'] == 'per_sample_mean':
                sample -= np.mean(sample, axis = 0).reshape((1, channels))
            if params['preprocess'] == 'per_sample_mean_variance':
                sample -= np.mean(sample, axis = 0).reshape((1, channels))
                sample /= (np.std(sample, axis = 0).reshape((1, channels)) + 1e-10)
            if params['preprocess'] == 'mean':
                sample -= params['mean'].reshape((1, channels))
            if params['preprocess'] == 'mean_variance':
                sample -= params['mean'].reshape((1, channels))
                sample /= params['std'].reshape((1, channels))
            if sample.shape[0] < length:
                sample = np.concatenate([np.zeros((length - sample.shape[0], channels),
                                                 'float32'),
                                         sample], axis = 0)
            x_chunk[j] = np.asarray(sample, 'float32').T.reshape((channels, 1, length))
            y_chunk[j, :] = labels[sub, series][t, events]

        yield x_chunk, y_chunk, chunk_length

###############################################################################################
def test_valid_chunk_gen_fun(data, labels, events, params):
    """
    test_valid_chunk_gen_fun: function to generate chunks for validation set with series-major sequencial order
    parameters:
        data: EEG data from which to generate the chunks
        labels: labels corresponding to the input data
        events: list of target events
        params: parameter dictionary for generation
    """
    channels = params['channels']
    length = params['length']
    num_events = len(events)
    chunk_size = params['chunk_size']

    ###############################################################################################
    interval = 1
    if 'interval' in params:
        interval = params['interval']

    num_subs = labels.shape[0]
    num_series = labels.shape[1]
    sample_cells = np.zeros((0, 3), 'int32')
    for series in np.arange(num_series):
        series_all_cells = np.zeros((0, 3), 'int32')
        for sub in np.arange(num_subs):
            l = labels[sub, series]
            series_cells = np.zeros((len(l), 3), 'int32')
            series_cells[:, 0] = sub
            series_cells[:, 1] = series
            series_cells[:, 2] = np.arange(len(l))
            series_all_cells = np.concatenate((series_all_cells, series_cells), axis = 0)
        sample_cells = np.concatenate((sample_cells, series_all_cells[::interval, :]), axis = 0)
    ###############################################################################################

    num_lens = 1
    lens = [length]
    if 'test_lens' in params:
        lens = params['test_lens']
        num_lens = len(lens)

    temp = np.zeros((0, 4), 'int32')
    for s_len in lens:
        temp = np.concatenate((temp,
                               np.concatenate((sample_cells, s_len * np.ones((len(sample_cells), 1), 'int32')), axis = 1)),
                              axis = 0)
    sample_cells = temp
    num_cells = len(sample_cells)
    num_chunks = int(np.ceil(float(num_cells) / chunk_size))
    last_chunk_length = np.mod(num_cells, chunk_size)
    if last_chunk_length == 0:
        last_chunk_length = chunk_size
    idcs_gen = lambda n: np.arange(n * chunk_size, (n + 1) * chunk_size)

    print "num_chunks is %d" % num_chunks
    for i in np.arange(num_chunks):
        idcs = idcs_gen(i)
        x_chunk = np.zeros((chunk_size, channels, 1, length), 'float32')
        y_chunk = np.zeros((chunk_size, num_events), 'int32')
        if i != num_chunks - 1:
            chunk_length = chunk_size
        else:
            chunk_length = last_chunk_length
            idcs[chunk_length:] = 0

        for j, idx in enumerate(idcs):
            sub = sample_cells[idx, 0]
            series = sample_cells[idx, 1]
            t = sample_cells[idx, 2]
            s_len = sample_cells[idx, 3]

            # Copy the history data into sample. The stop critirion for copy is smaller than t+1
            sample = np.copy(data[sub, series][max((0, t - s_len + 1)):t + 1, :])
            if params['preprocess'] == 'per_sample_mean':
                sample -= np.mean(sample, axis = 0).reshape((1, channels))
            if params['preprocess'] == 'per_sample_mean_variance':
                sample -= np.mean(sample, axis = 0).reshape((1, channels))
                sample /= (np.std(sample, axis = 0).reshape((1, channels)) + 1e-10)
            if params['preprocess'] == 'mean':
                sample -= params['mean'].reshape((1, channels))
            if params['preprocess'] == 'mean_variance':
                sample -= params['mean'].reshape((1, channels))
                sample /= params['std'].reshape((1, channels))
            if sample.shape[0] < s_len:
                sample = np.concatenate([np.zeros((s_len - sample.shape[0], channels),
                                                  'float32'),
                                         sample], axis = 0)
            if length != s_len:
                factor = float(length) / s_len
                #sample = zoom(sample, [factor, 1], order = 1)
                sample = inline_zoom(sample, length)

            x_chunk[j] = np.asarray(sample, 'float32').T.reshape((channels, 1, length))
            y_chunk[j, :] = labels[sub, series][t, events]

        yield x_chunk, y_chunk, chunk_length
###############################################################################################

def sequence_chunk_gen_fun(data, labels, events, params):
    """
    sequence_chunk_gen_fun: function to generate chunks for test set with sequencial order
    parameters:
        data: EEG data from which to generate the chunks
        labels: labels corresponding to the input data
        events: list of target events
        params: parameter dictionary for generation
    """
    channels = params['channels']
    length = params['length']
    num_events = len(events)
    chunk_size = params['chunk_size']

    num_subs = labels.shape[0]
    num_series = labels.shape[1]
    sample_cells = np.zeros((0, 3), 'int32')
    for sub in np.arange(num_subs):
        for series in np.arange(num_series):
            l = labels[sub, series]
            series_cells = np.zeros((len(l), 3), 'int32')
            series_cells[:, 0] = sub
            series_cells[:, 1] = series
            series_cells[:, 2] = np.arange(len(l))
            sample_cells = np.concatenate((sample_cells, series_cells), axis = 0)

    interval = 1
    if params['section'] is 'valid' and 'interval' in params:
        interval = params['interval']
    sample_cells = sample_cells[::interval, :]

    num_lens = 1
    lens = [length]
    if 'test_lens' in params:
        lens = params['test_lens']
        num_lens = len(lens)

    temp = np.zeros((0, 4), 'int32')
    for s_len in lens:
        temp = np.concatenate((temp,
                               np.concatenate((sample_cells, s_len * np.ones((len(sample_cells), 1), 'int32')), axis = 1)),
                              axis = 0)
    sample_cells = temp
    num_cells = len(sample_cells)
    num_chunks = int(np.ceil(float(num_cells) / chunk_size))
    last_chunk_length = np.mod(num_cells, chunk_size)
    if last_chunk_length == 0:
        last_chunk_length = chunk_size
    idcs_gen = lambda n: np.arange(n * chunk_size, (n + 1) * chunk_size)

    print "num_chunks is %d" % num_chunks
    for i in np.arange(num_chunks):
        idcs = idcs_gen(i)
        x_chunk = np.zeros((chunk_size, channels, 1, length), 'float32')
        y_chunk = np.zeros((chunk_size, num_events), 'int32')
        if i != num_chunks - 1:
            chunk_length = chunk_size
        else:
            chunk_length = last_chunk_length
            idcs[chunk_length:] = 0

        for j, idx in enumerate(idcs):
            sub = sample_cells[idx, 0]
            series = sample_cells[idx, 1]
            t = sample_cells[idx, 2]
            s_len = sample_cells[idx, 3]

            # Copy the history data into sample. The stop critirion for copy is smaller than t+1
            sample = np.copy(data[sub, series][max((0, t - s_len + 1)):t + 1, :])
            if params['preprocess'] == 'per_sample_mean':
                sample -= np.mean(sample, axis = 0).reshape((1, channels))
            if params['preprocess'] == 'per_sample_mean_variance':
                sample -= np.mean(sample, axis = 0).reshape((1, channels))
                sample /= (np.std(sample, axis = 0).reshape((1, channels)) + 1e-10)
            if params['preprocess'] == 'mean':
                sample -= params['mean'].reshape((1, channels))
            if params['preprocess'] == 'mean_variance':
                sample -= params['mean'].reshape((1, channels))
                sample /= params['std'].reshape((1, channels))
            if sample.shape[0] < s_len:
                sample = np.concatenate([np.zeros((s_len - sample.shape[0], channels),
                                                  'float32'),
                                         sample], axis = 0)
            if length != s_len:
                factor = float(length) / s_len
                #sample = zoom(sample, [factor, 1], order = 1)
                sample = inline_zoom(sample, length)

            x_chunk[j] = np.asarray(sample, 'float32').T.reshape((channels, 1, length))
            y_chunk[j, :] = labels[sub, series][t, events]

        yield x_chunk, y_chunk, chunk_length

def chunk_gen(chunk_gen_fun):
    #chunk_gen_fun就是train_net.py里的
    """
    getattr(data_, model.train_data_params['chunk_gen_fun'])(data[:, model.train_series],
                                                                 labels[:, model.train_series],
                                                                 model.events,
                                                                 model.train_data_params))
    就是data_.random_chunk_gen_fun()
    """
    #
    # chunk_gen_fun = data_.random_chunk_gen_fun()
    Q = Queue.Queue(maxsize = 1)
    """
    Constructor for a FIFO queue.
    maxsize sets the upperbound limit
    on the number of items that can be placed
    in the queue. Insertion will block
    once this size has been reached,
    until queue items are consumed.
    If maxsize is less than or equal to zero,
    the queue size is infinite.
    """

    def thread_fun(chunk_gen_fun, buff):
        #buff会用具体的Q，导入，即Queue
        for chunk in chunk_gen_fun:
            #chunk_gen_fun就是random_chunk_gen_fun()的返回值
            #其返回值是 yield x_chunk, y_chunk, chunk_size
            #返回的是多个值。怎么回事？？？？
            buff.put(chunk, block = True)
            """
            调用队列对象的put()方法在队尾插入一个项目。
            put()有两个参数，第一个item为必需的，为插入项目的值；
            第二个block为可选参数，默认为1。
            如果队列当前为空且block为1，
            put()方法就使调用线程暂停,直到空出一个数据单元。
            如果block为0，put方法将引发Full异常。
            """
        buff.put(None)

    thread = threading.Thread(target = thread_fun,
                              args = (chunk_gen_fun, Q))
    """
    创建一个线程
    class threading.Thread(group=None, target=None,
    name=None, args=(), kwargs={}, *, daemon=None)
    This constructor should always be called with
    keyword arguments. Arguments are:
    target,thread_fun(), is the callable object
    to be invoked by the run() method.

    args,(chunk_gen_fun, Q), is the argument tuple
    for the target invocation.
    """
    thread.daemon = True
    thread.start()
    """
    This is a case where threading is used as
    a simple optimization: each subthread
    is waiting for thread_fun() to process and respond,
    in order to put its contents on the queue;
    each thread is a daemon 'thread.daemon = True'(won't keep the process
    up if main thread ends -- that's more common
    than not); the main thread starts 'thread.start()'
    all subthreads, does a get 'Q.get' on the queue
    to wait until one of them
    has done a put 'buff.put(chunk, block = True)'
    then emits the results and terminates
    (which takes down any subthreads
    that might still be running,
    since they're daemon threads).

    Proper use of threads in Python is
    invariably connected to I/O operations
    (since CPython doesn't use multiple cores
    to run CPU-bound tasks anyway, the only reason
    for threading is not blocking the process
    while there's a wait for some I/O).
    Queues are almost invariably the best way
    to farm out work to threads and/or collect
    the work's results, by the way, and
    they're intrinsically threadsafe
    so they save you from worrying about
    locks, conditions, events, semaphores,
    and other
    inter-thread coordination/communication concepts.
    """

    for chunk in iter(Q.get, None):
        yield chunk
