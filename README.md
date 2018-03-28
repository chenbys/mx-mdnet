剔除离群bbox
多GPU有什么操作吗？
hard minibatch mining
不同层用不同学习率

抽样不使用img_pad
不update老旧的负样本

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

glove的frame8，pre_region=gt时，topK精准，但是probs不高：0.4,0.5左右；其他pre_region的准度不高但probs较低；说明overfitting?