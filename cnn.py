

from utils import *
from einops import rearrange
 
class Net(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, n, 3, padding=1, padding_mode='reflect')  # 32x32
        self.block1 = ResBlock(n, n)
        self.block2 = ResBlockDown(n, 2*n)   # 16x16
        self.block3 = ResBlock(2*n, 2*n)
        self.block4 = ResBlockDown(2*n, 3*n)   # 8x8
        self.block5 = ResBlock(3*n, 3*n)
        self.block6 = ResBlockDown(3*n, 4*n)   # 4x4
        self.block7 = ResBlock(4*n, 4*n)
        self.block8 = ResBlock(4*n, 4*n)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(4*4*4*n, 10)
        
    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = rearrange(x, 'b c h w -> b (c h w)')
        self.prelast = x 
       # x = self.dropout(x)
        x = self.dense(x)
        
        return x