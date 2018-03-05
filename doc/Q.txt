改loss，高iou比低iou将近2倍
剔除离群bbox
多GPU有什么操作吗？
hard minibatch mining
不同层用不同学习率

train_data与predict_data是否一致？
train_data是由N个patch构成

隐患：
1.image reshape HWC to CHW
2.mdnet_vot-otb.mat载入，conv1输出一致，lrn2输出小数点后三位有小误差。怀疑是精度误差，但误差应该可以无视。