addpath(genpath('/home/chenjunjie/workspace/MDNet_for_votkit'))
vl_setupnn
net=load('models/mdnet_vot-otb.mat')
rnet.layers=net.layers(1:1)
data=ones([219,227,3,1]);
data=gpuArray(single(data));
res=vl_simplenn(rnet,data,[],[]);
r=gather(res(end).x);
