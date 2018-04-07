多GPU有什么操作吗？
hard minibatch mining
不同层用不同学习率


抽样不使用img_pad,一个batch含1个img_patch，各个batch的img_patch的尺寸不用，该如何设置？
如果不同的话，new mx.io.NDArrayIter会出错


不update老旧的负样本

birds2 seq


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0
bolt1 338f
    会跟到另一个人，虽然两者的分都很高0.99.
剔除离群bbox!


刚开始对第0帧，1帧输出的分数比较低，0.6,0.7
随着更新，输出越来越高，到0.9,0.95。对第0帧的输出也变成了0.9了。
这是过拟合？

OOM,如何节省内存，为什么更新了8个epoch后，在第9个epoch中出OOM
for epoch in epoches:
    for batch in batches:
        mod.forward_backward(batch)
        mod.update()