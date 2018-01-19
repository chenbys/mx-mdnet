计划：
1.代价函数该用loss(pred,overlap),而不是loss(pred,{1,0})

待解决问题：
1.每一帧提取36个patch，每个patch200正例，200负例。可以增加。一般patch有正例400，负例3000

问题：
1.loss越来越大，pred出现nan，最终pred全是nan
2.过拟合严重

隐患：
1.image reshape HWC to CHW
2.mdnet_vot-otb.mat载入，conv1输出一致，lrn2输出小数点后三位有小误差。怀疑是精度误差，但误差应该可以无视。
