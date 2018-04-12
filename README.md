多GPU有什么操作吗？
hard minibatch mining
不同层用不同学习率

params init
MDNet直接load进来的
如何选择初始化?Uniform, Normal

不update老旧的负样本

mod.bind在pycharm中要花51s,无论是Debug还是run
而在vot-toolkit中3s


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0

剔除离群bbox!
loss降不下去，从0.68一般降到0.35，MDNet一般降到0.16或0.07左右。
还不能做到类内识别，和MDNet不一样，应该是网络结构改变带来的影响

但是PR和RR和TopKACC还行，0.99-0.8-0.99。
bolt1 338f
    会跟到另一个人，虽然两者的分都很高0.99.
    
刚开始对第0帧，1帧输出的分数比较低，0.6,0.7
随着更新，输出越来越高，到0.9,0.95。对第0帧的输出也变成了0.9了。
这是过拟合？

OOM,如何节省内存，为什么更新了8个epoch后，在第9个epoch中出OOM
for epoch in epoches:
    for batch in batches:
        mod.forward_backward(batch)
        mod.update()