计划：
1.代价函数该用loss(pred,overlap),而不是loss(pred,{1,0})

问题：
1.loss越来越大，pred出现nan，最终pred全是nan
2.过拟合严重

隐患：
1.image reshape HWC to CHW
2.mdnet_vot-otb.mat载入，conv1输出一致，lrn2输出小数点后三位有小误差。怀疑是精度误差，但误差应该可以无视。
