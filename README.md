剔除离群bbox
多GPU有什么操作吗？
hard minibatch mining
不同层用不同学习率

抽样不使用img_pad
不update老旧的负样本


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0


multi-tracking 用一个pred_iter
pre_region三倍w,h

track(model, plt.imread(config.img_paths[0]), config.gts[0], config.gts[0])
INFO:root:PR:0.88,RR:0.17,TopK:1.00,IOU:0.64

INFO:root:Epoch[4] Train-PR=0.902847
INFO:root:Epoch[4] Train-RR=0.907361
INFO:root:Epoch[4] Train-TrackTopKAcc=0.991556
INFO:root:Epoch[4] Time cost=1.037
INFO:root:Epoch[4] Validation-PR=0.963978
INFO:root:Epoch[4] Validation-RR=0.605833
INFO:root:Epoch[4] Validation-TrackTopKAcc=0.982222
也就是说，训练集上很好，预测集不好。训练集与预测集是否一致？ 
但是其中预测集包含0.5到0.7的样本，可能会扰乱PR,RR,Loss
ball1序列，说明预测集与训练集确实有些不一致，但是在pre_region=gt的情况下，topK还是比较准确的

bolt1序列，bbox一开始有点宽，然后逐渐变得宽，此时如果pre_region=gt的topK能检测出来窄的bbox，不过分数较低0.8多。
pre_region=pre_region的topK检测出来的bbox很宽分数有0.9多。
1.更相信以前的样本


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