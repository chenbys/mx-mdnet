net=load('mdnet_vot-otb.mat')
net=net.layers
conv1=net{1,1}
conv123.conv1_filters=gather(conv1.filters);
conv123.conv1_biases=gather(conv1.biases);
conv2=net{1,5}
conv123.conv2_filters=gather(conv2.filters);
conv123.conv2_biases=gather(conv2.biases);
conv3=net{1,9}
conv123.conv3_filters=gather(conv3.filters);
conv123.conv3_biases=gather(conv3.biases);
save conv123 conv123
