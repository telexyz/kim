![](docs/files/project.png)

## (h, w) kernel implementation
https://gist.github.com/tiendung/1f8fc03707da89139ad1508f2ca262dd

## pytorch max-pooling
https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

## 2x1 max-pool trick

```py
import kim as ndl

a = ndl.default_device().rand(5,2)
b = ndl.NDArray.make(a.shape, strides=(2,-1), handle=a._handle, offset=1).compact()
mask = (a > b)
a * mask
```

![](docs/files/project1.jpg)

![](docs/files/project2.jpg)