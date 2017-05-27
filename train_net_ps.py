"""
Train a single model via an RCNN

Usage:
    train_net.py <model_name> [<resume_file>] [<no_test>]

Options:
    <model_name>    Name of model file (without the folder prefix or .py suffix)
    <resume_file>   Path to cached resume file
    <no_test>       Flag to control test mode
"""

from __future__ import print_function

import cPickle
from importlib import import_module
import os
# import sys
import time
from itertools import izip
import warnings

import docopt#command line的接口表达功能
#要跟三个双引号配合一起用
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lasagne as nn
import numpy as np
import numpy.random as nr

import data as data_
#import data 应该是导入data.py这个程序

# import metrics
#注意：此处把metrics.py这个程序的导入注释掉了
#所以metrics.py不要了！

def main():
    """
    Main function for running from the command line.
    """

    # Parse command line
    args = docopt.docopt(__doc__)
    #docopt 是 Python 的一个第三方参数解析库，
    # 可以根据使用者提供的文档描述自动生成解析器。
    # 因此使用者可以用它来定义交互参数与解析参数
    #文档描述有两个部分：Usage和Option.
    #Usage: 以一个空行结束，冒号后面的第一个单词
    # 作为程序的名称。名称后面就是参数的描述，
    #注意:3个双引号里的内容有特殊意义，比如Usage就是关键字
    model_name = args['<model_name>']
    #model_name终结modelsfolder里是一个执行程序
    #但是这里不是，就仅仅是个名字
    #len3584_resize3_bs_c1r4p5_f9n256r35p1.py
    test_flag = args['<no_test>'] is None
    resume_file = args['<resume_file>']
    #resume_file文件很奇怪，只有调用的地方，
    #没有生成这个文件的命令！！！？？？

    # Echo arguments
    print('model_name: {}'.format(model_name))
    print('test_flag: {}'.format(test_flag))
    print('resume_file: {}'.format(resume_file))

    # Set random seed deterministically for reproducibility
    # nr.seed(int(time.time()))
    nr.seed(2016)

    # Create directory for caching partial results
    resume_path = os.path.join('models', model_name)
    #models 文件夹与里的model_name连接成一个路径
    if not os.path.exists(resume_path):
        os.mkdir(resume_path)
        #如果models文件夹里没有这个folder，那就创建一个

    # Resume training if specified
    if resume_file:
        #resume_file应该是先前生成的一些结果数据，
        #需要再次调用的。
        #在哪里生成的，那个文件里？格式是什么？
        if not os.path.exists(resume_file):
            raise Exception('Resume file {} does not exist'.format(resume_file))
        print('Resuming previous session...')
        resume_data = np.load(resume_file)
        exp_id = resume_data['exp_id']
        resume_path, _ = os.path.split(resume_file)
    else:
        exp_id = '%s' % time.strftime('%Y%m%d-%H%M%S', time.localtime())
        #Experiment ID
        resume_path = os.path.join('models', model_name, exp_id)
        if not os.path.exists(resume_path):
            os.mkdir(resume_path)
            #如果resume_file不存在，那么创建一个时间戳为名称的folder

    # Build the model
    print()
    print("Experiment ID: %s" % exp_id)
    print()

    print("Building model...")
    model = import_module('models.%s' % model_name)
    #%s代表字符串变量，就是model_nam
    #from importlib import import_module
    #动态导入模块，import_module只是简单地执行和import相同的步骤，
    # 但是返回生成的模块对象。你只需要将其存储在一个变量，
    # 然后像正常的模块一样使用。
    #注意：import进来的就一定是程序了。不是名字
    #所以此时model句柄就是len3584xxx.py
    #models是folder，model_name是Python的程序名。
    #注意此处文件夹与文件的关系不用\斜杠。而是.点号！
    l_out = model.build_model()
    #根据model name构建model
    #比如：build_model()就是models文件夹里len3584_resize3_bs_c1r4p5_f9n256r35p1.py
    #model子程序里的方法
    #l_out是构建好的模型的句柄
    x_shared, y_shared, idx, lr, iter_train, iter_valid = \
        model.build_train_valid(l_out)
    #lr = theano.shared(np.float32(lr_schedule(0)))
    #Theano的shared类似于全局变量的概念，
    # 其值将会在多个函数中共用。
    #shared变量lr
    #此处为做完train和valid后的输出结果
    chunk_idx = 0
    #什么是chunk？？
    #chunk不是episode就是batch。到底是哪个？
    #不是batch, batch = 64
    #好像是chunk的总数，num_chunks = 400
    #每个chunk的size，可能是指包括的数据条数为，
    # chunk_size = 4096
    #总共400个chunk包含的数据条数，即time stamp数，为
    #4096 X 400 = 1,638,400
    #每个series大概有20万条数据。每人有8个series，总共12人
    #那么总数据条数为：
    #20万 X 8 X 12 = 19,200,000
    #也就是说从总数据19,200,000条里随机取出了
    #1,638,400条数据来训练
    #也就是说：
    # 1,638,400条数据切成了400个chunk
    #每个chunk4096 条数据
    #每个batch64个数据。
    #所以每个chunk 的episode是：4096 / 64 = 64, episode = 64
    #400个chunk总共有episode: 64 X 400 = 25600个
    metrics = model.metrics
    # 注意：model.metrics指的是，model里的metrics列表
    # metrics = [metrics.meanAccuracy, metrics.meanAUC]
    # 不是指：导入的是根目录下的metrics.py程序
    metric_names = model.metric_names
    # 注意：
    # model.metrics_names指的是，model里的metrics_names列表
    #metric_names = ['mean accuracy', 'areas under the ROC curve']
    #
    losses_train = []
    scores_train = []
    scores_valid = []
    best_record = [0, 0]

    # Resume history information if neccessary
    if 'resume_data' in dir():
        print('resume')
        nn.layers.set_all_param_values(l_out, resume_data['param_values'])
        chunk_idx = resume_data['chunk_idx']
        losses_train = resume_data['losses_train']
        scores_train = resume_data['scores_train']
        scores_valid = resume_data['scores_valid']
        best_record = resume_data['best_record']
        data_.neg_pool = resume_data['neg_pool']

    chunk_idcs = np.arange(chunk_idx, model.train_data_params['num_chunks'])
    #train_data_params就是len3584xxx.py里面的元组参数集
    #其中'num_chunks': 400, chunk_idx = 0
    #相当于chunk_idcs=[0:400]

    # Load data, and get the data generation functions for each section
    data, labels = data_.load(model.data_path)
    #调用model里的data_path变量
    #model其实就是len3584_resize3_bs_c1r4p5_f9n256r35p1.py
    #其实就是data_path = 'eeg_train.npy'
    #注意此处同时导入data,和 labels，不是仅仅labels

    #!!!
    # # exclude the second series of second subject
    # data[1, 1] = np.zeros([0, 32], 'float32')
    # labels[1, 1] = np.zeros([0, 6], 'float32')
    #!!!

    # for subject-specific training
    #data = data[model.subjects, :][:]
    #labels = labels[model.subjects, :][:]

    data_.init_sample_cells(labels, model.events, model.train_series,
                            model.valid_series)
    #data_就是当前目录下的data.py程序
    #调用data.py下的相应参数
    #init_sample_cells()就是把所有有动作的发生的time stamp
    #的索引号和没有动作的发生的time stamp的索引号
    #都挑出来。有动作的发生的都放在pos_cells里,
    # 没有动作的发生的都放在neg_cells
    """
    train_series = [0, 1, 2, 3, 4, 5]
    valid_series = [6, 7]
    在train data中，每个sub有8个series
    前6个series做训练集。后2个series做校验集
    每个series中大致有21万个数据
    总数据量
    210Kx8x12=20.16M
    我的数据量：
    10年股票，每年220天交易
    总共5000只
    5Kx10x220=11M
    根这个测试集基本在同一个数量级，可以用
    test_series = [0, 1, 2, 3, 4, 5]
    events = [0, 1, 2, 3, 4, 5]#6个动作
    num_events = len(events)
    """
    """
    init_sample_cells: get all the positive and
    negative indices for train series and
    valid series
    parameters:
        labels: ground truth of all the data for train and valid
        events: list of target events for training
        train_series: list of series for training
        valid_series: list of series for validation
    """
    # 总共有8个series。把头6个series做training data
    #train_series = [0, 1, 2, 3, 4, 5]
    #把后2个最validation data
    #valid_series = [6, 7]
    #test_series = [0, 1, 2, 3, 4, 5]
    #event是指那6个动作
    #events = [0, 1, 2, 3, 4, 5]

    train_data_gen = lambda: data_.chunk_gen(
        getattr(data_, model.train_data_params['chunk_gen_fun'])(data[:, model.train_series],
                                                                 labels[:, model.train_series],
                                                                 model.events,
                                                                 model.train_data_params))
    # 'chunk_gen_fun': 'random_chunk_gen_fun',
    #注意此处train_data_gen()是一个函数。
    #lambda定义的是一个函数，不是一个变量
    #到时候直接调用train_data_gen()就可以了。
    """
    这个lambda其实是train_data_gen =
    lambda: data_.chunk_gen(getattr()())
    data_.chunk_gen()才是lambda的真正函数。
    这个lambda没有自变量。所以冒号：前是空的
    getattr()()两个括号并排，看上去奇怪，其实后面的
    括号就是前面getattr()构造出的新函数的参数！
    getattr()构造出的新函数是：data_.random_chunk_gen_fun()
    random_chunk_gen_fun()函数在data.py里的定义是：
    def random_chunk_gen_fun(data, labels, events, params):
    所以这里定义的4个参数就对应着下面的四个参数！！！
    (
    data[:, model.train_series],
    labels[:, model.train_series],
    model.events,
    model.train_data_params)
    )
    这里其实是一个嵌套的函数：
    data_.chunk_gen(data_.random_chunk_gen_fun())
    然后把这个嵌套函数的句柄通过lambda赋值给了：
    train_data_gen() -> data_.chunk_gen()
    """
    valid_data_gen = lambda: data_.chunk_gen(
        getattr(data_, model.valid_data_params['chunk_gen_fun'])(data[:, model.valid_series],
                                                                 labels[:, model.valid_series],
                                                                 model.events,
                                                                 model.valid_data_params))

    bs_data_gen = lambda: data_.chunk_gen(
        getattr(data_, model.bs_data_params['chunk_gen_fun'])(data[:, model.train_series],
                                                              labels[:, model.train_series],
                                                              model.events,
                                                              model.bs_data_params))

    do_validation = True
    if 'test_valid' in model.test_data_params and model.test_data_params['test_valid'] == True:
        do_validation = True
    else:
        do_validation = False

    valid_result_folder = 'model_combine/all_valid_results/'
    test_result_folder = 'model_combine/all_test_results/'

    # Start training
    very_start = time.time()
    for chunk_idx, (x_chunk, y_chunk, _) in izip(chunk_idcs, train_data_gen()):
        #train_data_gen()的返回结果就是 chunk，
        #在data.py中用由chunk_gen(chunk_gen_fun)计算得出
        """
        izip的使用
        izip(*iterator)
        [python] view plain copy print?
        import itertools
        listone = ['a','b','c']
        listtwo = ['11','22','abc']
        for item in itertools.izip(listone,listtwo):
            print item,
结果：('a', '11') ('b', '22') ('c', 'abc')
功能：返回迭代器，项目是元组，元组来自*iterator的组合
        """


        start_time = time.time()
        lr.set_value(model.lr_schedule(chunk_idx))
        #lr就是theano.shared
        #set_value 用来重新设置参数，
        x_shared.set_value(x_chunk)
        #x_shared = nn.utils.shared_empty(dim = len(input_dims))
        #lasagne.utils.shared_empty(dim=2, dtype=None)[source]
        #Creates empty Theano shared variable.
        #with the specified number of dimensions
        #set_value()方法在这里有点特殊，因为utils.shared_empty()
        #没有这个方法
        y_shared.set_value(y_chunk)

        if chunk_idx == chunk_idcs[0]:
            losses = []
            preds_train = np.zeros((0, model.num_events), 'float32')
            y_train = np.zeros((0, model.num_events), 'int32')

        y_train = np.concatenate([y_train, y_chunk], axis=0)
        num_batches_chunk = model.train_data_params[
            'chunk_size'] / model.batch_size
        #batch_size = 64
        #chunk_size = 4096
        #num_batches_chunk = 4096 / 64
        #num_batches_chunk = 64
        for b in np.arange(num_batches_chunk):
            loss, pred = iter_train(b)
            """
           iter_train = theano.function([idx], [train_loss, train_output],
                                 givens = givens,
                                 updates = updates)
            """
            if np.isnan(loss):
                raise RuntimeError("NaN Detected.")
            losses.append(loss)
            preds_train = np.concatenate([preds_train, pred], axis=0)

        if ((chunk_idx + 1) % model.display_freq) == 0:
            print()
            print("Chunk %d/%d, lr = %.7f" % (
                chunk_idx + 1, model.train_data_params['num_chunks'],
                lr.get_value()))

            mean_train_loss = np.mean(losses)
            print("  mean training loss:\t\t%.6f" % mean_train_loss)
            losses_train.append(mean_train_loss)

            scores = [chunk_idx + 1]
            for i, metric in enumerate(metrics):
                scores.append(metric(y_train, preds_train))
                print("  %s:" % metric_names[i])
                print(scores[-1])

            scores_train.append(scores)
            print("  The best score is %f, obtained in %d chunks" % (
                best_record[1], best_record[0]))
            end_time = time.time()
            print("  elapsed time is %f seconds" % (end_time - start_time))
            print("  system time is ", time.strftime('%Y%m%d-%H%M%S',
                                                     time.localtime()))
            print("  elapsed time from the begining is %f seconds" %
                end_time - very_start)
            losses = []
            preds_train = np.zeros((0, model.num_events), 'float32')
            y_train = np.zeros((0, model.num_events), 'int32')

        if ((chunk_idx + 1) % model.valid_freq) == 0 and do_validation is True:
            print()
            print("Evaluating valid set")
            start_time = time.time()
            preds_valid = np.zeros((0, model.num_events), 'float32')
            y_valid = np.zeros((0, model.num_events), 'int32')
            for x_chunk, y_chunk, chunk_length in valid_data_gen():
                y_valid = np.concatenate(
                    [y_valid, y_chunk[:chunk_length, :]], axis=0)
                num_batches = int(np.ceil(chunk_length / float(model.batch_size)))
                x_shared.set_value(x_chunk)
                chunk_output = np.zeros((0, model.num_events), 'float32')
                for b in np.arange(num_batches):
                    pred = iter_valid(b)
                    chunk_output = np.concatenate((chunk_output, pred), axis=0)
                chunk_output = chunk_output[:chunk_length, :]
                preds_valid = np.concatenate((preds_valid, chunk_output), axis=0)

            scores = [chunk_idx + 1]
            for i, metric in enumerate(metrics):
                scores.append(metric(y_valid, preds_valid))
                print("  %s:" % metric_names[i])
                print(scores[-1])

            scores_valid.append(scores)
            if best_record[1] < scores[-1][-1]:
                best_record[0] = chunk_idx + 1
                best_record[1] = scores[-1][-1]
            print("  The best score is %f, obtained in %d chunks" % (
                best_record[1], best_record[0]))
            end_time = time.time()
            print("  elapsed time is %f seconds" % (end_time - start_time))
            print()

        if ((chunk_idx + 1) % model.bs_freq == 0) and (
                chunk_idx != chunk_idcs[-1]):
            print()
            print("Bootstrap the training set")
            start_time = time.time()
            preds_bs = np.zeros((0, model.num_events), 'float32')
            y_bs = np.zeros((0, model.num_events), 'int32')
            for x_chunk, y_chunk, chunk_length in bs_data_gen():
                y_bs = np.concatenate([y_bs, y_chunk[:chunk_length, :]], axis=0)
                num_batches = int(np.ceil(chunk_length / float(model.batch_size)))
                x_shared.set_value(x_chunk)
                chunk_output = np.zeros((0, model.num_events), 'float32')
                for b in np.arange(num_batches):
                    pred = iter_valid(b)
                    chunk_output = np.concatenate((chunk_output, pred), axis=0)
                chunk_output = chunk_output[:chunk_length, :]
                preds_bs = np.concatenate((preds_bs, chunk_output), axis=0)

            scores = [chunk_idx + 1]
            for i, metric in enumerate(metrics):
                scores.append(metric(y_bs, preds_bs))
                print("  %s:" % metric_names[i])
                print(scores[-1])
            data_.bootstrap(y_bs, preds_bs)

            end_time = time.time()
            print("  elapsed time is %f seconds" % (end_time - start_time))
            print()

        if ((chunk_idx + 1) % model.save_freq) == 0:
            print()
            print("Saving model")
            save_path = os.path.join(resume_path, '%d' % (chunk_idx + 1))
            with open(save_path, 'w') as f:
                cPickle.dump({
                    'model': model_name,
                    'exp_id': exp_id,
                    'chunk_idx': chunk_idx + 1,
                    'losses_train': losses_train,
                    'scores_train': scores_train,
                    'scores_valid': scores_valid,
                    'best_record': best_record,
                    'param_values': nn.layers.get_all_param_values(l_out),
                    'neg_pool': data_.neg_pool
                }, f, cPickle.HIGHEST_PROTOCOL)

    # Test all valid and save results
    if 'test_valid' in model.test_data_params and model.test_data_params['test_valid'] == True:
        data, labels = data_.load(model.data_path)
        test_valid_data_gen = lambda: data_.chunk_gen(
            getattr(data_, model.test_valid_params['chunk_gen_fun'])(data[:, model.valid_series],
                                                                     labels[:, model.valid_series],
                                                                     model.events,
                                                                     model.test_valid_params))
        print()
        print("Testing all valid samples")
        start_time = time.time()
        preds_test = np.zeros((0, model.num_events), 'float32')
        y_test = np.zeros((0, model.num_events), 'int32')
        idx = 1
        for x_chunk, y_chunk, chunk_length in test_valid_data_gen():
            # t1 = time.time()
            y_test = np.concatenate([y_test, y_chunk[:chunk_length, :]], axis=0)
            num_batches = int(np.ceil(chunk_length / float(model.batch_size)))
            x_shared.set_value(x_chunk)
            chunk_output = np.zeros((0, model.num_events), 'float32')
            for b in np.arange(num_batches):
                pred = iter_valid(b)
                chunk_output = np.concatenate((chunk_output, pred), axis=0)
            chunk_output = chunk_output[:chunk_length, :]
            preds_test = np.concatenate((preds_test, chunk_output), axis=0)
            # t2 = time.time()
            idx += 1
        save_valid_name = model_name[:-2] + str(model.valid_series[0]) + str(
            model.valid_series[1])
        save_path = os.path.join(valid_result_folder,
                                 'test_valid_' + save_valid_name + '.npy')
        np.save(save_path, [y_test, preds_test])
        end_time = time.time()
        print("  elapsed time is %f seconds" % (end_time - start_time))

    # Test
    if ((not model.test_data_params.has_key('test')) or
            model.test_data_params['test'] == True) and test_flag is True:
        data, labels = data_.load('eeg_test.npy')
        model.test_data_params['section'] = 'test'
        test_data_gen = lambda: data_.chunk_gen(
            getattr(data_, model.test_data_params['chunk_gen_fun'])(data,
                                                                    labels,
                                                                    model.events,
                                                                    model.test_data_params))
        print()
        print("Testing")
        start_time = time.time()
        preds_test = np.zeros((0, model.num_events), 'float32')
        y_test = np.zeros((0, model.num_events), 'int32')
        idx = 1
        for x_chunk, y_chunk, chunk_length in test_data_gen():
            # t1 = time.time()
            y_test = np.concatenate([y_test, y_chunk[:chunk_length, :]], axis=0)
            num_batches = int(np.ceil(chunk_length / float(model.batch_size)))
            x_shared.set_value(x_chunk)
            chunk_output = np.zeros((0, model.num_events), 'float32')
            for b in np.arange(num_batches):
                pred = iter_valid(b)
                chunk_output = np.concatenate((chunk_output, pred), axis=0)
            chunk_output = chunk_output[:chunk_length, :]
            preds_test = np.concatenate((preds_test, chunk_output), axis=0)
            # t2 = time.time()
            idx += 1

        save_path = os.path.join(test_result_folder, 'test_' + model_name + '.npy')
        np.save(save_path, [y_test, preds_test])

    end_time = time.time()
    print("  elapsed time is %f seconds" % (end_time - start_time))

if __name__ == '__main__':
    main()
