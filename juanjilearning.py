import  urllib.request
import  os
import tarfile
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
tf.reset_default_graph()
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
#定义标签字典，每一个数字所代表的图像类别的名称
label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"fog",7:"horse",8:"ship",9:"truck"}
#定义显示图像数据及其对应标签的函数
def show(images,labels,prediction,idx,num=10):
    flg=plt.gcf()
    flg.set_size_inches(12,6)
    if num>10:
        num=10
    for i in range(0,num):
        ax=plt.subplot(2,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title=str(i)+','+label_dict[labels[idx]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[idx]]
        ax.set_title(title,fontsize=10)
        idx+=1
    plt.show()
#show(Xtest,Ytest,[],0,10)
#图像数字标准化
Xtrain_nomalize=Xtrain.astype('float32')/255.0
Xtest_nomalize=Xtest.astype('float32')/255.0
#标签数据处理
encoder=OneHotEncoder(sparse=False)
yy=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
encoder.fit(yy)
Ytrain_reshape=Ytrain.reshape(-1,1)
Ytrain_onehot=encoder.transform(Ytrain_reshape)
Ytest_reshape=Ytest.reshape(-1,1)
Ytest_onehot=encoder.transform(Ytest_reshape)
#定义共享函数
def weight(shape):
    #构建模型师，需要使用tf.Variable来创建一个变量，训练时，这个变量不断更新，使用函数。。。normaled生成标准差为0.1的随机数来初始化权值
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name="W")
def bias(shape):
    #定义偏置，初始化为0.1
    return tf.Variable(tf.constant(0.1,shape=shape),name='b')
def conv2d(x,W):
    #卷积操作，步长为1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    #定义池化操作，步长为2，及原尺寸的长和宽除以2
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#输入层
#32x32图像，通道为3（RGB）
with tf.name_scope('input_layer'):
    x=tf.placeholder('float',shape=[None,32,32,3],name='X')
#第1个卷积层，输入通道为3，输出通道32，卷积后图像尺寸不变，依然是32x32
with tf.name_scope('conv_1'):
    W1=weight([3,3,3,32])#卷积核的宽，卷积核的高，输入通道，输出通道
    b1=bias([32])
    conv_1=conv2d(x,W1)+b1
    conv_1=tf.nn.relu(conv_1)
#第1个池化层，将32x32的图像缩小为16X16，池化不改变通道数量依然是32
with tf.name_scope('pool_1'):
    pool_1=max_pool_2x2(conv_1)
#第2个卷积层，输入通道32，输出通道64，卷积后尺寸不变，依然是16x16
with tf.name_scope('conv_2'):
    W2=weight([3,3,32,64])
    b2=bias([64])
    conv_2=conv2d(pool_1,W2)+b2
    conv_2=tf.nn.relu(conv_2)
#第2个池化层，将16x16的图像缩小为8x8，池化不改变通道数量，依然 是64
with tf.name_scope('pool_2'):
    pool_2=max_pool_2x2(conv_2)
#全连接层，将第二个池化层的64个8X8的图像转换为一维的向量，长度是64*8*8=4096
#128个神经元
with tf.name_scope('fc'):
    W3=weight([4096,128])
    b3=bias([128])
    flat=tf.reshape(pool_2,[-1,4096])
    h=tf.nn.relu(tf.matmul(flat,W3)+b3)
    h_dropout=tf.nn.dropout(h,keep_prob=0.8)
#输出层，共有十个神经元，对应0-9这10个类别
with tf.name_scope('output_layer'):
    W4=weight([128,10])
    b4=bias([10])
    pred=tf.nn.softmax(tf.matmul(h_dropout,W4)+b4)
#构建模型
with tf.name_scope("optimizer"):
    y=tf.placeholder("float",shape=[None,10],name='label')
    #定义损失函数
    loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    #选择优化器
    optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.0001).minimize(loss_function)
#定义准确率
with tf.name_scope("evaluation"):
    correct_prediction=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

from time import time
train_epochs=25
batch_size=50
total_batch=int(len(Xtrain)/batch_size)
epoch_list=[]
accuracy_list=[]
loss_list=[]
epoch=tf.Variable(0,name='epoch',trainable=False)
starttime=time()
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

ckpt_dir=r"E:\log\\"
saver=tf.train.Saver(max_to_keep=1)
summary_writer=tf.summary.FileWriter(ckpt_dir,sess.graph)
ckpt=tf.train.latest_checkpoint(ckpt_dir)
if ckpt!=None:
    saver.restore(sess,ckpt)#加载所有参数
    #从这里开始就可以直接使用模型进行预测，或者继续训练了
else:
    print("Training from scratch")
start=sess.run(epoch)
print("Training starts from {} epoch".format(start+1))
#迭代训练
def get_train_batch(number,batch_size):
    return Xtrain_nomalize[number*batch_size:(number+1)*batch_size],Ytrain_onehot[number*batch_size:(number+1)*batch_size]
for ep in range(start,train_epochs):
    for i in range(total_batch):
        batch_x,batch_y=get_train_batch(i,batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if i%100==0:
            print("Step {}".format(i),"finished")
    loss,acc=sess.run([loss_function,accuracy],feed_dict={x:batch_x,y:batch_y})
    epoch_list.append(ep+1)
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train epoch:","%2d"%(sess.run(epoch)+1),"Loss=","{:.6f}".format(loss),"Accuracy=",acc)
    saver.save(sess,ckpt_dir+"",global_step=ep+1)
    sess.run(epoch.assign(ep+1))
duration=time()-starttime
print("Train finished task:",duration)
