
from layers import *


class Net(nn.Module):
    def __init__(self, n, nh, nout=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, n//2, 5, padding=2, stride=2, padding_mode='reflect')
        #self.conv2 = nn.Conv2d(n//2, n, 3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(n//2, n//2, 3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(n//2, n, 3, padding=1, stride=2, padding_mode='reflect')
        self.predense = nn.Linear(4*4*3, n)
        
        self.posenc = PositionalEncoding2d(n)

        self.seed = Seed(n, 1)
        self.enc1 = EncoderBlock(n, n, nh)
        self.enc2 = EncoderBlock(n, n, nh)
        self.enc3 = EncoderBlock(n, n, nh)
        self.enc4 = EncoderBlock(n, n, nh)
        self.enc5 = EncoderBlock(n, n, nh)
        self.enc6 = EncoderBlock(n, n, nh)

        self.dense = nn.Linear(n, nout)
        
        self.dropout = nn.Dropout(0.1)
        self.cuda()
        
    def forward(self, x):
        '''
        x = rearrange(x, 'b h w c -> b c h w')
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = rearrange(x, 'b c h w -> b h w c')'''
        x = rearrange(x, 'b (h f1) (w f2) c -> b h w (f1 f2 c)', f1=4, f2=4)
        x = relu(self.predense(x))
        
        x = self.posenc(x)
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.dropout(x)

        y = self.seed(x) # classifier token
        x = torch.cat((y, x), 1)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)

        x = x[:,0,:]
        self.prelast = x
        x = self.dense(x)
        
        return x