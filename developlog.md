# dataprovider

实现了一个dataset，能够读入csv数据，根据size进行滑动窗口切割

> 记住常用包的位置：dataset和dataloader在torch.utils.data下
>
> 还有pandas的用法，读出csv后，怎么获取行(.column)，怎么裁剪指定行/列(第一维是行，第二维是列)，怎么增加/删除行(drop(column=, axis=))
>
> map函数怎么用(.map(函数表达式))
>
> lambda表达式怎么理解

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



# main

我们希望事情是这样的：加载数据->训练模型->保存模型->预测输出结果

所以先加载数据，并且建立dataloader

然后建立模型model

然后建立训练器trainer

> Trainer在包pytorch_lightning.trainer下
