# spMat

稀疏矩阵的保存和右乘向量，简单的实现。同样暂时不关心速度/效率。

为了能够方便地用 A*b 的操作，矩阵使用Compressed Row Storage(CRS)格式或者称为 Compressed sparse row(CSR)格式储存。

# 说明

使用Eigen的结果作为比对测试。

程序需要读取文件格式说明如下，如果需要diy测试可以自行准备：

**矩阵数据格式**

spA: 对称稀疏方阵
```
rows cols
r0 c0 v0
r1 c1 v1
...
rn cn vn
```

b: 列向量
```
v1
v2
...
vn
```

# reference

[wiki](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))