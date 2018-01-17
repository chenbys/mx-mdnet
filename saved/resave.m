net=load('mdnet_otb-vot15.mat')
net=net.layers
conv1=net{1,1}
conv1_filters=gather(conv1.filters);
conv1_biases=gather(conv1.biases);
conv2=net{1,5}
conv2_filters=gather(conv2.filters);
conv2_biases=gather(conv2.biases);
conv3=net{1,9}
conv3_filters=gather(conv3.filters);
conv3_biases=gather(conv3.biases);
save conv123 conv123
