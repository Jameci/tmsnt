# dataprovider

实现了一个dataset，能够读入csv数据，根据size进行滑动窗口切割

关于怎么做normalization，我纠结了一段时间，原因是timesnet代码中使用了train部分的数据去得到了scaler，之后用这个scaler去transform了全部的数据，无论输入数据x还是输出数据y

这不太符合常识，所以我决定走一条不一样的路，我只拿train部分的scaler去transform输入x，对输出y不做处理

> 正则化可以用sklearn.preprocessing中的StandardScaler来做，其用法也很简单
>
> 建立一个scaler = StandardScaler()
>
> 之后scaler.fit(x_train)取得训练数据集的均值和方差
>
> 然后scaler.transform(x)利用训练集的均值和方差对整个数据集进行标准化
>
> 值得注意的是fit方法对于每个scaler只能被执行一次，如果想要直接用一个数据集的均值方差去transform它自己，可以使用fit_transform方法

# timesnetmodel

第一步是先有一个框架

> 使用pl.LightningModel的话，需要实现的三个方法是：init，trainingstep和configure_optimizers

## init

> py_lightning的模型继承自pytorch_lightning.LightningModule
>
> 调用super时，传入super()中的参数分别是类名和self，其它和构建父类有关的参数传入super的init函数里

## trainingstep

需要返回一个loss，看了原代码里用的是MSEloss

> mesloss函数在torch.nn.functional里，一般这个包被引用时会as F

然后另起了一个model，看到原代码里用了block，也准备了一个block的框架在这里

> 一般模型需要继承torch.nn.model

forward是nn.model用于前向传播的关键函数，由于我不知道该怎么利用y，所以只取了x和x_mark作为输入

有了forward，trainingstep就可以填坑了

### nn.model

#### forward

在正式填坑之前，想到先用一个线性层测试一下

> nn.Linear()只能在一个维度上进行处理，如果输入是高维的话，必须保证输入的最后一维和给出的输入size一致

验证通过了……

看了一下第一步是要对输入进行normalization，真的需要吗，我先省了

##### embedding

第二步是embedding，看了一下embedding所需要的函数，好多啊

仔细分析下，embedding所调用的是Data_Embedding，一共做了四件事

用value_embedding，temporal_embedding，position_embedding处理了x和x_mark

将上述处理结果相加并dropoout

所有的embedding都是把输入数据从(batch, seq, feature)变为(batch, seq, dmodel)

feature是输入特征的维度，dmodel默认是512，这里好像改成了64

###### value_embedding

仅仅是把x拿过来，在时间维度上做了一个一维的卷积，我也会

###### position_embedding

position返回的值只和输入的尺寸有关，和输入的具体值无关，具体原因写在飞书文档里了

###### temporal_embedding

对x_mark作的，用的是position_embedding的方法

最后实现一个大的embedding，把它们合在一起

## optimizer

直接使用最经典的torch.optim.Adam(self.parameters(), lr=1e-3)

> self.parameters()是怎么获取参数的？其实也比较简单——就遍历一下模型的子模块就可以了

# main

我们希望事情是这样的：加载数据->训练模型->保存模型->预测输出结果

所以先加载数据，并且建立dataloader

然后建立模型model

然后建立训练器trainer

> Trainer在包pytorch_lightning.trainer下
