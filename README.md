剔除离群bbox
多GPU有什么操作吗？
hard minibatch mining
不同层用不同学习率


抽样不使用img_pad,一个batch含1个img_patch，各个batch的img_patch的尺寸不用，该如何设置？
如果不同的话，new mx.io.NDArrayIter会出错


不update老旧的负样本


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0



INFO:root:Epoch[0] Time cost=1.052
INFO:root:@CHEN-> IOU : [ 0.66 ] !!!  prob: 0.80 for tracking on frame 40, cost 5.9870
INFO:root:@CHEN-> IOU : [ 0.63 ] !!!  prob: 0.85 for tracking on frame 41, cost 0.4083
INFO:root:@CHEN-> IOU : [ 0.68 ] !!!  prob: 0.89 for tracking on frame 42, cost 1.0273
INFO:root:@CHEN-> IOU : [ 0.72 ] !!!  prob: 0.89 for tracking on frame 43, cost 0.3689
INFO:root:@CHEN-> IOU : [ 0.79 ] !!!  prob: 0.89 for tracking on frame 44, cost 1.0248
[16:09:27] src/operator/././cudnn_algoreg-inl.h:107: Running performance tests to find the best convolution algorithm, this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
INFO:root:@CHEN-> IOU : [ 0.72 ] !!!  prob: 0.90 for tracking on frame 45, cost 0.3702
INFO:root:@CHEN-> IOU : [ 0.98 ] !!!  prob: 0.89 for tracking on frame 46, cost 0.1865
INFO:root:@CHEN-> IOU : [ 0.75 ] !!!  prob: 0.87 for tracking on frame 47, cost 0.1811
INFO:root:@CHEN-> IOU : [ 0.56 ] !!!  prob: 0.85 for tracking on frame 48, cost 0.1728
INFO:root:@CHEN-> IOU : [ 0.45 ] !!!  prob: 0.80 for tracking on frame 49, cost 0.2023

INFO:root:Epoch[0] Time cost=1.163
INFO:root:@CHEN-> IOU : [ 0.66 ] !!!  prob: 0.75 for tracking on frame 10, cost 4.6887
INFO:root:@CHEN-> IOU : [ 0.62 ] !!!  prob: 0.82 for tracking on frame 11, cost 0.2509
INFO:root:@CHEN-> IOU : [ 0.65 ] !!!  prob: 0.78 for tracking on frame 12, cost 0.2522
INFO:root:@CHEN-> IOU : [ 0.64 ] !!!  prob: 0.76 for tracking on frame 13, cost 0.2801
INFO:root:@CHEN-> IOU : [ 0.44 ] !!!  prob: 0.67 for tracking on frame 14, cost 0.2778
INFO:root:@CHEN-> IOU : [ 0.53 ] !!!  prob: 0.62 for tracking on frame 15, cost 0.2889
INFO:root:@CHEN-> IOU : [ 0.54 ] !!!  prob: 0.77 for tracking on frame 16, cost 0.2818
INFO:root:@CHEN-> IOU : [ 0.47 ] !!!  prob: 0.71 for tracking on frame 17, cost 0.2731
INFO:root:@CHEN-> IOU : [ 0.68 ] !!!  prob: 0.69 for tracking on frame 18, cost 0.2806
INFO:root:@CHEN-> IOU : [ 0.79 ] !!!  prob: 0.73 for tracking on frame 19, cost 0.2759