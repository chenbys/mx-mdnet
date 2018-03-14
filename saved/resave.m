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
fc4=net{1,11}
conv123.fc4_filters=gather(fc4.filters);
conv123.fc4_biases=gather(fc4.biases);
fc5=net{1,14}
conv123.fc5_filters=gather(fc5.filters);
conv123.fc5_biases=gather(fc5.biases);
save conv123fc4fc5 conv123 
