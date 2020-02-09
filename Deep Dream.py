from __future__ import print_function
import os
from io import BytesIO
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf
import numpy as np
import cv2 as cv

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
model_fn = r'E:\\inc\\tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn,'rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input=tf.placeholder(np.float32,name='input')
#图像预处理——减少均值
imagenet_mean=117.0#在训练inc模型时候做了减少均值预处理，此处有人需要减同样的均值保持一致
#图像预处理——增加维度，图像的数据格式一般是高宽通道数，为了同时将多张图片输入网络而在前面增加一维，变为batch高宽通道数，多了一个批次里样本的数量
t_preprocessed=tf.expand_dims(t_input-imagenet_mean,0)#减去均值并且增加维度
#导入模型，并将经预处理的图像送入网络中
tf.import_graph_def(graph_def,{'input':t_preprocessed})
#找出卷积层
layers=[op.name for op in graph.get_operations()if op.type=='Conv2D' ]
print(len(layers))
#name1='mixed4d_3x3_bottleneck_pre_relu'
#print('shape of %s:%s'%(name1,str(graph.get_tensor_by_name('import/'+name1+':0').get_shape())))

def savearray(img_array,img_name):
    cv.imwrite("E:\\temp\\"+img_name,img_array)#保存图像文件
    print("img saved:%s"%img_name)
def  render_naive(t_obj,img0,iter_n=20,step=1.0):
    t_score=tf.reduce_mean(t_obj)
    t_grad=tf.gradients(t_score,t_input)[0]
    img=img0.copy()
    for i in range(iter_n):
        g,score=sess.run([t_grad,t_score],{t_input:img})
        g/=g.std()+1e-8
        img+=g*step
        print("iter:%d" % (i+1),"score(mean)=%f" % score)
    savearray(img,"naive_deepdreamall.jpg")
# name='mixed4c'
# layer_output=graph.get_tensor_by_name("import/%s:0"%name)
# print(layer_output)
# img_test=PIL.Image.open(r"C:\Users\23102\Desktop\mountain.jpg")
# img_noise=np.random.uniform(size=(244,244,3))+100.0
# render_naive(layer_output,img_noise,iter_n=100)

#通过单通道特征生成deepdream图像
# name2='mixed4d_3x3_bottleneck_pre_relu'
# channel=139
#print('shape of %s:%s'%(name2,str(graph.get_tensor_by_name('import/'+name2+':0').get_shape())))
# lay_output=graph.get_tensor_by_name("import/%s:0"%name2)
# img_noise=np.random.uniform(size=(244,244,3))+100.0
# render_naive(lay_output[:,:,:,channel],img_noise,iter_n=20)
# #im=PIL.Image.open("E:\\temp\\naive_deepdream.jpg")
# #im.show()
# #im.save("E:\\temp\\mountain.jpg")
# im=cv.imread("E:\\temp\\naive_deepdream2.jpg")
# cv.imshow('im',im)
# cv.waitKey()

#较低层单通道卷积特征生成Deepdream图像
# name3='mixed3a_3x3_bottleneck_pre_relu'
# lay_output=graph.get_tensor_by_name("import/%s:0"%name3)
# img_noise=np.random.uniform(size=(244,244,3))+100.0
# channel=86
#print('shape of %s:%s'%(name3,str(graph.get_tensor_by_name('import/'+name3+':0').get_shape())))
# render_naive(lay_output[:,:,:,channel],img_noise,iter_n=20)
# im=cv.imread("E:\\temp\\naive_deepdream3.jpg")
# cv.imshow('im',im)
# cv.waitKey()

# #较高层单通道卷积特征生成Deepdream图像
# name4="mixed5b_5x5_pre_relu"
# lay_output=graph.get_tensor_by_name("import/%s:0"%name4)
# img_noise=np.random.uniform(size=(244,244,3))+100.0
# channel=118
# print('shape of %s:%s'%(name4,str(graph.get_tensor_by_name('import/'+name4+':0').get_shape())))
# render_naive(lay_output[:,:,:,channel],img_noise,iter_n=20)
# im=PIL.Image.open("E:\\temp\\naive_deepdream4.jpg")
# im.show()
# #im.save("E:\\temp\\mountain.jpg")

# #所有通道
# name="mixed4d_3x3_bottleneck_pre_relu"
# lay_output=graph.get_tensor_by_name("import/%s:0"%name4)
# img_noise=np.random.uniform(size=(244,244,3))+100.0
# channel=118
# print('shape of %s:%s'%(name4,str(graph.get_tensor_by_name('import/'+name4+':0').get_shape())))
# render_naive(lay_output,img_noise,iter_n=20)
# im=PIL.Image.open("E:\\temp\\naive_deepdreamall.jpg")
# im.show()
def calc_grad_tiled(img,t_grad,title_size=512):
    sz=title_size
    h,w=img.shape[:2]
    sx,sy=np.random.randint(sz,size=2)
    img_shift=np.roll(np.roll(img,sx,1),sy,0)
    grad=np.zeros_like(img)
    for y in range(0,max(h-sz//2,sz),sz):
        for x in range(0,max(w-sz//2,sz),sz):
            sub=img_shift[y:y+sz,x:x+sz]
            g=sess.run(t_grad,{t_input:sub})
            grad[y:y+sz,x:x+sz]=g
    return np.roll(np.roll(grad,-sx,1),-sy,0)
def resize(img,hw):
    min=img.min()
    max=img.max()
    img=(img-min)/(max-min)*255
    img=np.float32(scipy.misc.imresize(img,hw))
    img=img/255*(max-min)+min
    return img
def render_deepdream(t_obj,img0,iter_n=10,step=1.5,octave_n=4,octave_scale=1.4):
    #后两个表示金字塔层数已经层与层之间的倍数
    t_score=tf.reduce_mean(t_obj)
    t_grad=tf.gradients(t_score,t_input)[0]
    img=img0.copy()
    octaves=[]
    for i in range(octave_n-1):
        hw=img.shape[:2]
        lo=resize(img,np.int32(np.float32(hw)/octave_scale))
        hi=img-resize(lo,hw)
        img=lo
        octaves.append(hi)
    for octave in range(octave_n):
        if octave > 0:
            hi=octaves[-octave]
            img=resize(img,hi.shape[:2])+hi
        for i in range(iter_n):
            g=calc_grad_tiled(img,t_grad)
            img+=g*(step/(np.abs(g).mean()+1e-7))
    img=img.clip(0,255)
    savearray(img,'mountain_deepdream.jpg')
    im=PIL.Image.open("E:\\temp\\mountain_deepdream.jpg").show()

name='mixed4c'
layer_output=graph.get_tensor_by_name("import/%s:0"%name)
img0=PIL.Image.open(r"C:\Users\23102\Desktop\mountain.jpg")
img0=np.float32(img0)
render_deepdream(tf.square(layer_output),img0)
