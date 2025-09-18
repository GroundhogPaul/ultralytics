# 源代码

代码路径：ultralytics\ultralytics\nn\modules\block.py.Bottleneck

```python
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

# 用法及理念1：细瓶口，减少计算量和过拟合

默认的e=0.5即细瓶口

典型用法是通过1x1卷积减少隐藏层ch数，从而降低隐藏层3x3卷积的计算量

由于降维的存在，也可以起到减少计算量的效果

# 用法及理念2：粗瓶口

详见C2
