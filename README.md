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
    
刚开始对第0帧，1帧输出的分数比较低，0.6,0.7
随着更新，输出越来越高，到0.9,0.95。对第0帧的输出也变成了0.9了。
这是过拟合？

OOM,如何节省内存，为什么更新了8个epoch后，在第9个epoch中出OOM
for epoch in epoches:
    for batch in batches:
        mod.forward_backward(batch)
        mod.update()
        
        
我又不怕死地做了pre_train的代码，发现wd不能设太高．
在第一帧训练里面，如果wd是5e-4的话，正样本输出都是0.9999999了，太恐怖了．
设为6e0才能看起来正常地输出0.9,0.8之类的．
wd设高了不是据说防过拟合嘛．