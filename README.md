速度分析：安装cudnn
精度分析：bbox reg
失败率分析：vot-tookit评测，0 iou is failure and shut down even come back in subsequence.

不update老旧的负样本

[14:49:45] src/operator/././cudnn_algoreg-inl.h:107: Running performance tests to find the best convolution algorithm, this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0

    
刚开始对第0帧，1帧输出的分数比较低，0.6,0.7
随着更新，输出越来越高，到0.9,0.95。对第0帧的输出也变成了0.9了。
这是过拟合？

        
我又不怕死地做了pre_train的代码，发现wd不能设太高．
在第一帧训练里面，如果wd是5e-4的话，正样本输出都是0.9999999了，太恐怖了．
设为6e0才能看起来正常地输出0.9,0.8之类的．
wd设高了不是据说防过拟合嘛．
其实更高的样本检测的出来，只是有些不准确的，更高一点点的分．

pre_train什么时候算收敛

参数怎么设的合适，pre_train wd:5e-4,tracking:1.5e0
能够很好地过bolt1,ball1,但是birds2和fish4不太好．

hnm做了，效果还行

imresize和feat sample耗时多导致要50ms，实际上predict只需要10ms．总的说来，需要80ms
