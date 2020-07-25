<br><br>
<font size=5>**项目目录**</font>
<br><br>
&emsp;&emsp;<font size=5>1.简单介绍Pix2pix和CyecleGAN</font>
<br><br>
&emsp;&emsp;<font size=5>2.训练前的准备</font>
<br><br>
&emsp;&emsp;<font size=5>3.开始模型的训练</font>
<br><br>
&emsp;&emsp;<font size=5>4.预测</font>
<br><br><br><br>

<font size=5>**1.简单介绍Pix2pix和CyecleGAN**</font>
<br><br>

&emsp;&emsp;&emsp;&emsp;<font size=4>**1.1.简单的认识并了解GAN**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>可以从该博客中简单了解GAN的原理：[博客](https://blog.csdn.net/on2way/article/details/72773771)</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>**1.2.简单的认识并了解Pix2pix**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>Pix2pix论文地址：[“Image-to-Image Translation with Conditional Adversarial Networks”](https://arxiv.org/pdf/1611.07004.pdf)</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>可以从该博客中简单了解Pix2pix的原理：[博客](https://blog.csdn.net/stdcoutzyx/article/details/78820728)</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>**1.3.简单的认识并了解CyecleGAN**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>CyecleGAN论文地址：[“Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”](https://arxiv.org/abs/1703.10593)</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>可以从该博客中简单了解CyecleGAN的原理：[博客](https://blog.csdn.net/qq_21190081/article/details/78807931)</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>**1.4.Pix2pix与CyecleGAN对比**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>pix2pix必须使用成对的数据进行训练，即可以看作是一 一映射的。</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://ai-studio-static-online.cdn.bcebos.com/6867664b70ab4ddc9a6970576a6a097ee75470dc2699488a93a960dcb0b2b12c" style="zoom:110%">
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>但在现实生活中一 一对应的数据是很难获取到的，比如下面的图，我们想把一处风景变成某种风格的油画，现实生活中是很少有与数据集中该风景相对应的油画的（哪有那么多的画家嘛QAQ）：</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>因此CyecleGAN就是为了解决这样一个问题，就是训练集不一定需要在两个域(domain)中有完全配对的图，只需要两种模式不同的图即可。</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://ai-studio-static-online.cdn.bcebos.com/983e208bb20d4026882648853eae9482fba1a2d3fd5c4530be42b95c199fdb82" style="zoom:110%">
<br><br>

<font size=5>**2.训练前的准备**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>**2.1.解压数据集**</font>
<br>


```python
# QAQ

# 准备Pix2pix的数据集
!unzip data/data45935/dataset.zip -d /home/aistudio/dataset2/

# 准备CycleGan的数据集
!unzip data/data45935/dataset.zip -d /home/aistudio/dataset1/
!mv /home/aistudio/dataset1/dataset/* /home/aistudio/dataset1/
!rmdir /home/aistudio/dataset1/dataset/
```

&emsp;&emsp;&emsp;&emsp;<font size=4>**2.2.下载model**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3>将飞桨的GAN项目从GitHub上git下来：[GAN传送门](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/gan)</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>**2.3.生成训练时所需要的文件**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3>训练时所需要的文件作者已经准备好啦，就等你来运行了</font>
<br>


```python
# Pix2pix
# 训练时所需要的文件作者已经准备好啦，就等你来运行了

f = open('dataset2/train_list.txt','w')
for k in range(1,3001):
    f.write('trainA/'+str(k)+'.jpg'+'\t'+'trainB/'+str(k)+'.jpg'+'\n')
f.close()
f = open('dataset2/test_list.txt','w')
for k in range(39991,39991+10):
    f.write('testA/'+str(k)+'.jpg'+'\t'+'testB/'+str(k)+'.jpg'+'\n')
f.close()
```


```python
# CycleGan
# 训练时所需要的文件作者已经准备好啦，就等你来运行了

f = open('dataset1/trainA.txt','w')
for k in range(1,2001):
    f.write('trainA/'+str(k)+'.jpg'+'\n')
f.close()
f = open('dataset1/trainB.txt','w')
for k in range(1,2001):
    f.write('trainB/'+str(k)+'.jpg'+'\n')
f.close()
f = open('dataset1/testA.txt','w')
for k in range(39991,39991+10):
    f.write('testA/'+str(k)+'.jpg'+'\n')
f.close()
f = open('dataset1/testB.txt','w')
for k in range(39991,39991+10):
    f.write('testB/'+str(k)+'.jpg'+'\n')
f.close()
```

<font size=5>**3.开始模型的训练**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>友情提醒：作者已经训练好了，可以直接用的，大家如果需要训练，请在训练的时候，尽量一边训练一边删除checkpoints和训练时网络生成的图片（会保存在output/pix2pix(cyclegan)/test目录下），不然会内存爆炸QAQ。</font>
<br><br>


```python
#安装imageio ,scipy
!pip install -q imageio
!pip install -q scipy==1.2.1
```

&emsp;&emsp;&emsp;&emsp;<font size=4>**3.1.Pix2pix训练**</font>
<br><br>


```python
!python gan/train.py --model_net Pix2pix \
                        --dataset dataset/ \
                        --data_dir dataset2/ \
                        --train_list dataset2/train_list.txt \
                        --test_list dataset2/test_list.txt \
                        --crop_type Random \
                        --dropout True \
                        --gan_mode vanilla \
                        --batch_size 1 \
                        --epoch 115 \
                        --image_size 286 \
                        --crop_size 256 \
                        --output ./output/pix2pix/
```

&emsp;&emsp;&emsp;&emsp;<font size=4>**3.2.CycleGAN训练**</font>
<br><br>


```python
!python gan/train.py --model_net CycleGAN \
                        --dataset /home/aistudio/dataset1 \
                        --batch_size 1 \
                        --net_G resnet_9block \
                        --g_base_dim 32 \
                        --net_D basic \
                        --norm_type batch_norm \
                        --epoch 150 \
                        --image_size 286 \
                        --crop_size 256 \
                        --crop_type Random \
                        --output ./output/cyclegan/

```

<font size=5>**4.预测**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>**4.1.Pix2pix模型**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=4>tips：生成的checkpoint文件太大了挂载不上去，请见谅QAQ</font>


```python
# 生成的checkpoint文件太大了挂载不上去，请见谅

!python gan/infer.py --init_model output/pix2pix/checkpoints/110/ \
                        --dataset_dir /home/aistudio/ \
                        --image_size 256 \
                        --n_samples 1 \
                        --crop_size 256 \
                        --model_net Pix2pix \
                        --net_G unet_256 \
                        --test_list /home/aistudio/test_list.txt \
                        --output ./infer_result/pix2pix/
```

&emsp;&emsp;&emsp;&emsp;<font size=4>**4.2.CycleGAN模型**</font>
<br><br>


```python
!python gan/infer.py --init_model output/cyclegan/checkpoints/48/ \
                        --dataset_dir /home/aistudio/ \
                        --image_size 256 \
                        --n_samples 1 \
                        --crop_size 256 \
                        --input_style A \
                        --test_list /home/aistudio/test_list.txt \
                        --model_net CycleGAN \
                        --net_G resnet_9block \
                        --g_base_dims 32 \
                        --output ./infer_result/cyclegan/

```

&emsp;&emsp;&emsp;&emsp;<font size=4>**4.3.Pix2pix模型结果可视化**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>原图</font>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=4>预测结果</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/593d543c06274c7f947a39386af86462d807205b86934fce891159bb8b623414)
&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/82e91e32b4664c00bab565b343decf60d0a0d2c77f43456487dca4d3285b1809)

&emsp;&emsp;&emsp;&emsp;<font size=4>**4.4.CycleGAN模型结果可视化**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>原图</font>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=4>预测结果</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/c49c708287bb48afabeb6c75b8c38709a7b7cd8714f84687aac891e4519a72df)
&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/dc8c6e5826b7453598d293d73caba92e6f974a6a5f1b42c6bb90925d5a0bebcc)
<br><br>
