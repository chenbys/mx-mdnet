计划：
0. 为什么，输出的loss对高iou label的样本更难拟合，结果是高iou label差异0.2,低差异0.1。
    方法：在loss函数上对高iou label的样本进行加权。例如，loss=|pred-label|*label. 这样行不行？
    这样只需要考虑梯度？更新label=0.5的样本比label=1的样本，可以看作学习率为只有一般，更不在意低iou label样本的死活？
    
0。多GPU有什么操作吗？
1. 代价函数该用CE(pred,overlap)
2. debug IOU_loss, 核对梯度和更新值等.可以用Monito，不过很奇异，label_re_0应该是label符号reshape出来的，名字上多了个_0，值还面目全非，
smooth_l1_0还有负数。不知道是什么情况！
3. hard minibatch mining, 不同层用不同学习率
4. MD训练。查看MDNet论文不MD训练和MD训练的精度。如何解决过拟合。

待解决问题：
1.每一帧提取36个patch，每个patch200正例，200负例。可以增加。一般patch有正例400，负例3000

问题：
1.过拟合严重,一般是train0.99,val0.6

隐患：
1.image reshape HWC to CHW
2.mdnet_vot-otb.mat载入，conv1输出一致，lrn2输出小数点后三位有小误差。怀疑是精度误差，但误差应该可以无视。
# TypeError: save only accept dict str->NDArray or list of NDArray