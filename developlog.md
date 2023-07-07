# dataprovider

实现了一个dataset，能够读入csv数据，根据size进行滑动窗口切割

> 记住常用包的位置：dataset和dataloader在torch.utils.data下
>
> 还有pandas的用法，读出csv后，怎么获取行，怎么裁剪指定行/列，怎么增加/删除行
>
> map函数怎么用
>
> lambda表达式怎么理解

# timesnetmodel

第一步是先有一个框架

> 使用pl.LightningModel的话，需要实现的三个方法是：init，trainingstep和configure_optimizers

# main

我们希望事情是这样的：加载数据->训练模型->保存模型->预测输出结果

所以先加载数据，并且建立dataloader
