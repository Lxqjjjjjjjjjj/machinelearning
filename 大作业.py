import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import csv
# 查看当前tensorflow的版本
print("当前tensorflow版本", tf.__version__)

# 【1 导入Fashion MNIST数据集】
'''
加载数据集将返回四个NumPy数组：
train_images和train_labels数组是训练集 ，即模型用来学习的数据。
针对测试集 ， test_images和test_labels数组对模型进行测试 
'''
'''
图像是28x28 NumPy数组，像素值范围是0到255。 标签是整数数组，范围是0到9。这些对应于图像表示的衣服类别 ：
标签	    类
0	    T恤
1	    裤子
2	    套衫/卫衣
3	    连衣裙
4	    外衣/外套
5	    凉鞋
6	    衬衫
7	    运动鞋
8	    袋子
9	    短靴/脚踝靴
'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 每个图片都映射到一个标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 【2 探索数据】
print("训练集总图片数：", train_images.shape)
print("训练集中标签数:", len(train_labels))
print("标签取值：", train_labels)
print("测试集总图片数：", test_images.shape)
print("测试集标签数：", len(test_labels))
print("标签取值：", test_labels)
# 【3 预处理数据】
# 在训练网络之前，必须对数据进行预处理。
# 如果检查训练集中的第九张图像，将
# 看到像素值落在0到255的范围内
plt.figure()
plt.imshow(train_images[8])#训练第九张图片
plt.colorbar()
plt.grid(False)
plt.show()
#为了验证数据的格式正确，让我们显示训练集中的前25张图像，并在每张图像下方显示类别名称。
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
#
# 将这些值缩放到0到1的范围，然后再将其输入神经网络模型。为此，将值除以255。以相同的方式预处理训练集和测试集非常重要：
train_images = train_images / 255.0
test_images = test_images / 255.0
#展示测试集的图片
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
plt.show()
# 【4 建立模型】
# 建立神经网络需要配置模型的各层，然后编译模型
# 搭建神经网络结构 神经网络的基本组成部分是层 。图层（神经网络结构）从输入到其中的数据中提取表示
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

'''
编译模型
在准备训练模型之前，需要进行一些其他设置。这些是在模型的编译步骤中添加的：
损失函数 -衡量训练期间模型的准确性。您希望最小化此功能，以在正确的方向上“引导”模型。
优化器 -这是基于模型看到的数据及其损失函数来更新模型的方式。
指标 -用于监视培训和测试步骤。以下示例使用precision ，即正确分类的图像比例。
'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#训练模型
model.fit(train_images, train_labels, epochs=10)#迭代十次
# 比较模型在训练数据集上的表现
train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=2)
print('\nTrain accuracy:', train_acc)
# 作出预测 通过测试模型，可以使用它来预测某些图像。模型的线性输出logits 。附加一个softmax层，以将logit转换为更容易解释的概率。
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(train_images)
print(predictions[8])
print(np.argmax(predictions[8]))

# 【5 测试模型】
model.fit(test_images, test_labels, epochs=10)#迭代十次
# 比较模型在测试数据集上的表现
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
# 作出预测 通过测试模型，可以使用它来预测某些图像。模型的线性输出logits 。附加一个softmax层，以将logit转换为更容易解释的概率。
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
#创建一个列表来存放结果
list=[]
#利用for循环对测试集里10000条图片进行预测
for i in range(10000):
    print(predictions[i])#这里会输出10个数据分别对应该图片对10个标签的把握
    print(np.argmax(predictions[i]))#选出最大的把我即为该图片对应的标签
    print('------------------------------------------------------------------------')
    # 写入多行用writerows
    max=np.argmax(predictions[i])
    list.append([i,max])

with open("result.csv","w",newline='') as csvfile:
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerow(["picture","lable"])
    #写入多行用writerows
    writer.writerows(list)






