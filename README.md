剔除离群bbox
多GPU有什么操作吗？
hard minibatch mining
不同层用不同学习率


抽样不使用img_pad,一个batch含1个img_patch，各个batch的img_patch的尺寸不用，该如何设置？
如果不同的话，new mx.io.NDArrayIter会出错


不update老旧的负样本

birds2 seq


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


bolt1 seq:
add update data :50~70ms
multi tracking :100ms~200ms
online update 300 : 0.9s
online update 75:0.22s

每次update后，第一次track就会慢很多？？？ 惰性计算？
INFO:root:| long term update
INFO:root:@CHEN->update 315.
INFO:root:Update[3685]: now learning rate arrived at 6.00000e-06, will not change in the future
INFO:root:| epoch 0, cost:0.7692
INFO:root:| online update, cost:0.769321
INFO:root:@CHEN-> IOU : [ 0.24 ] !!!  prob: 0.55 for tracking on frame 40, cost 1.4733
INFO:root:@CHEN->| tracking 42.jpg
INFO:root:| 2.146296， time for track
INFO:root:| 0.022857， time for track
INFO:root:| 0.025527， time for track
INFO:root:| 0.028258， time for track
INFO:root:| 0.027375， time for track

INFO:root:@CHEN->| tracking 50.jpg
INFO:root:| multi track, cost:0.195079
INFO:root:| add update data, cost:0.051098
INFO:root:@CHEN-> IOU : [ 0.66 ] !!!  prob: 0.69 for tracking on frame 49, cost 0.2466
INFO:root:@CHEN->| tracking 51.jpg
INFO:root:| multi track, cost:0.197912
INFO:root:| add update data, cost:0.049997
INFO:root:| long term update
INFO:root:@CHEN->update 375.
INFO:root:| epoch 0, cost:1.1210
INFO:root:| online update, cost:1.121157
INFO:root:@CHEN-> IOU : [ 0.52 ] !!!  prob: 0.66 for tracking on frame 50, cost 1.6952
INFO:root:@CHEN->| tracking 52.jpg
INFO:root:| multi track, cost:2.693559
INFO:root:| add update data, cost:0.056952
INFO:root:@CHEN-> IOU : [ 0.58 ] !!!  prob: 0.77 for tracking on frame 51, cost 2.7510
INFO:root:@CHEN->| tracking 53.jpg
INFO:root:| multi track, cost:0.206585
INFO:root:| add update data, cost:0.052129
INFO:root:@CHEN-> IOU : [ 0.69 ] !!!  prob: 0.86 for tracking on frame 52, cost 0.2592

INFO:root:@CHEN->| tracking 220.jpg
INFO:root:| multi track, cost:0.199109
INFO:root:| add update data, cost:0.051470
INFO:root:@CHEN-> IOU : [ 0.51 ] !!!  prob: 0.95 for tracking on frame 219, cost 0.2511
INFO:root:@CHEN->| tracking 221.jpg
INFO:root:| multi track, cost:0.203108
INFO:root:| add update data, cost:0.053687
INFO:root:| long term update
INFO:root:@CHEN->update 750.
INFO:root:Update[14737]: now learning rate arrived at 6.00000e-06, will not change in the future
INFO:root:| epoch 0, cost:2.2078
INFO:root:| online update, cost:2.207968
INFO:root:@CHEN-> IOU : [ 0.52 ] !!!  prob: 0.91 for tracking on frame 220, cost 3.0783
INFO:root:@CHEN->| tracking 222.jpg
INFO:root:| multi track, cost:5.171348
INFO:root:| add update data, cost:0.051169
INFO:root:@CHEN-> IOU : [ 0.63 ] !!!  prob: 0.89 for tracking on frame 221, cost 5.2230
INFO:root:@CHEN->| tracking 223.jpg
INFO:root:| multi track, cost:0.199412
INFO:root:| add update data, cost:0.050603
INFO:root:@CHEN-> IOU : [ 0.55 ] !!!  prob: 0.87 for tracking on frame 222, cost 0.2505

INFO:root:@CHEN-> IOU : [ 0.42 ] !!!  prob: 0.65 for tracking on frame 228, cost 0.2515
INFO:root:@CHEN->| tracking 230.jpg
INFO:root:| multi track, cost:0.195232
INFO:root:| add update data, cost:0.052494
INFO:root:@CHEN-> IOU : [ 0.58 ] !!!  prob: 0.87 for tracking on frame 229, cost 0.2482
INFO:root:@CHEN->| tracking 231.jpg
INFO:root:| multi track, cost:0.199516
INFO:root:| add update data, cost:0.051782
INFO:root:| long term update
INFO:root:@CHEN->update 750.
INFO:root:Update[15351]: now learning rate arrived at 6.00000e-06, will not change in the future
INFO:root:| epoch 0, cost:2.1995
INFO:root:| online update, cost:2.199727
INFO:root:@CHEN-> IOU : [ 0.46 ] !!!  prob: 0.76 for tracking on frame 230, cost 3.0572
INFO:root:@CHEN->| tracking 232.jpg
INFO:root:| multi track, cost:5.245644
INFO:root:| add update data, cost:0.050716
INFO:root:@CHEN-> IOU : [ 0.42 ] !!!  prob: 0.84 for tracking on frame 231, cost 5.2968
INFO:root:@CHEN->| tracking 233.jpg
INFO:root:| multi track, cost:0.202996
INFO:root:| add update data, cost:0.049993
INFO:root:@CHEN-> IOU : [ 0.45 ] !!!  prob: 0.85 for tracking on frame 232, cost 0.2534