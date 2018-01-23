计划：
1.代价函数该用CE(pred,overlap)

待解决问题：
1.每一帧提取36个patch，每个patch200正例，200负例。可以增加。一般patch有正例400，负例3000

问题：
1.过拟合严重,一般是train0.99,val0.6

隐患：
1.image reshape HWC to CHW
2.mdnet_vot-otb.mat载入，conv1输出一致，lrn2输出小数点后三位有小误差。怀疑是精度误差，但误差应该可以无视。
