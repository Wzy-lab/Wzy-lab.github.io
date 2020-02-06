import  urllib.request
import  os
import tarfile
import pickle as p
import numpy as np
filepath=r"E:\data\cifar-10-python.tar.gz"
if os.path.isfile(filepath):
    print('Date file already exist.')
if not os.path.isfile(filepath):
    tfile=tarfile.open(r"E:\data\cifar-10-python.tar.gz",'r:gz')
    result=tfile.extractall(r"E:\data\\")
    print('Extracted to E:/data/cifar-10-batches=py/')
else:
    print("Directory already exist.")
def load_CIFAR_batch(filename):
    #读取一个批次的样本
    with open(filename,'rb') as f:
        data_dict=p.load(f,encoding='bytes')
        images=data_dict[b'data']
        labels=data_dict[b'labels']
        #把原始数据调整为：BCWH
        images=images.reshape(10000,3,32,32)
        images=images.transpose(0,2,3,1)
        labels=np.array(labels)
        return images,labels
def load_CIFAR_data(data_dir):
    images_train=[]
    labels_train=[]
    for i in range(5):
        f=os.path.join(data_dir,'data_batch_%d'%(i+1))
        print('loading',f)
        #调用load_CIFAR_batch()获得批量的图像及其对应的标签
        image_batch,label_batch=load_CIFAR_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        Xtrain=np.concatenate(images_train)
        Ytrain=np.concatenate(labels_train)
        del image_batch,label_batch
    Xtest,Ytest=load_CIFAR_batch(os.path.join(data_dir,'test_batch'))
    print("Finished loadding CIFAR-10 date")
    #返回训练集和测试集的图像和标签
    return Xtrain,Ytrain,Xtest,Ytest
data_dir=r'E:/date/cifar-10-batches-py/'
Xtrain,Ytrain,Xtest,Ytest=load_CIFAR_data(data_dir)
#显示数据集信息
print('training date shape:',Xtrain.shape)
print('training labels shape:',Xtrain.shape)
print('test date shape:',Xtrain.shape)
print('test labels shape:',Xtrain.shape)