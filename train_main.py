import mxnet as mx
import numpy as np
import logging
import time
from mnist import mnist_loader
from symbol import mdnet
from dataset.datahelper import Datahelper
from train import train_kit
from kit import *

K, N = 2, 5
batch_size = 100
datahelper = Datahelper('mnist', batch_size)

data = mx.sym.var('data')
shared = mdnet.add_shared_part(data)
shared_arg_list = shared.list_arguments()[1:]
shared_aux_list = shared.list_auxiliary_states()

# new all branches
symbols = list()
prefixes = list()
models = list()

for domain_idx in range(0, N):
    prefix = 'branch' + str(domain_idx) + '_'
    prefixes.append(prefix)
    branch = mdnet.add_branch_part(shared, prefix=prefix)
    symbols.append(branch)
    model = mx.mod.Module(symbol=branch, context=mx.gpu(),
                          label_names=('branch' + str(domain_idx) + '_softmax_label',))
    model.bind([datahelper.get_data_desc()], label_shapes=[datahelper.get_label_desc(domain_idx)])
    model.init_params()
    models.append(model)

# for each pass of dataset
logging.getLogger().setLevel(logging.DEBUG)
for k in range(0, K):
    p('round:' + str(k), DEBUG)
    # for each domain
    for domain_idx in range(0, N):
        p('domain:' + str(domain_idx), DEBUG)
        # get cur_branch prefix
        prefix = prefixes[domain_idx]
        # get cur_branch
        cur_model = models[domain_idx]
        # get pre_branch
        pre_model = models[domain_idx - 1]
        # get the train_iter
        train_iter = datahelper.get_train_iter(domain_idx, k)
        # change domain
        arg_params, aux_params = pre_model.get_params()
        cur_model.set_params(arg_params, aux_params, allow_missing=True, force_init=True,
                             allow_extra=True)
        # fit
        cur_model.fit(train_iter, optimizer='sgd', optimizer_params={'learning_rate': 0.1},
                      eval_metric='acc', num_epoch=1)

# acc = mx.metric.Accuracy()
# model1.score(val_iter1, acc)
# print acc.get()
