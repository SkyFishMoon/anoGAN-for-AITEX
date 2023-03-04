import torch
import torch.nn as nn
import torchvision.transforms as transforms


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.h, self.w = size
        self.mha = nn.MultiheadAttention(channels, 4)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.h * self.w).transpose(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.transpose(2, 1).view(-1, self.channels, self.h, self.w)


class Generator(nn.Module):
	def __init__(self, z_dim, c_dim, gf_dim):
		super(Generator, self).__init__()
		self.trans0 = nn.ConvTranspose2d(z_dim, gf_dim*8, 4, 1, 0, bias=False)
		self.bn0 = nn.BatchNorm2d(gf_dim*8)
		self.relu0 = nn.ReLU(inplace=True)
		
		self.trans1 = nn.ConvTranspose2d(gf_dim*8, gf_dim*4, 4, 2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(gf_dim*4)
		self.relu1 = nn.ReLU(inplace=True)
		
		self.trans2 = nn.ConvTranspose2d(gf_dim*4, gf_dim*2, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(gf_dim*2)
		self.relu2 = nn.ReLU(inplace=True)
			
		self.trans3 = nn.ConvTranspose2d(gf_dim*2, gf_dim*2, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(gf_dim*2)
		self.relu3 = nn.ReLU(inplace=True)
		
		self.trans4 = nn.ConvTranspose2d(gf_dim*2, gf_dim*2, 4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(gf_dim*2)
		self.relu4 = nn.ReLU(inplace=True)

		self.trans5 = nn.ConvTranspose2d(gf_dim*2, gf_dim, 4, 2, 1, bias=False)
		self.bn5 = nn.BatchNorm2d(gf_dim)
		self.relu5 = nn.ReLU(inplace=True)

		self.trans6 = nn.ConvTranspose2d(gf_dim, c_dim, 4, 2, 1, bias=False)

		self.tanh = nn.Tanh()

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0.0, 0.02)
				if m.bias is not None:
					m.bias.data.zero_()


	def forward(self, input):
		h0 = self.relu0(self.bn0(self.trans0(input)))
		h1 = self.relu1(self.bn1(self.trans1(h0)))
		h2 = self.relu2(self.bn2(self.trans2(h1)))
		h3 = self.relu3(self.bn3(self.trans3(h2)))
		h4 = self.relu4(self.bn4(self.trans4(h3)))
		h5 = self.relu5(self.bn5(self.trans5(h4)))
		h6 = self.trans6(h5)

		output = self.tanh(h6)
		return output # (c_dim, 64, 64)


class Discriminator(nn.Module):
	def __init__(self, c_dim, df_dim):
		super(Discriminator, self).__init__()
		self.conv0 = nn.Conv2d(c_dim, df_dim, 4, 2, 1, bias=False)
		self.relu0 = nn.LeakyReLU(0.2, inplace=True)
		self.att0 = SelfAttention(df_dim, (128, 128))
		
		self.conv1 = nn.Conv2d(df_dim, df_dim*2, 4, 2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(df_dim*2)
		self.relu1 = nn.LeakyReLU(0.2, inplace=True)
		self.att1 = SelfAttention(df_dim*2, (64, 64))
		
		self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(df_dim*4)
		self.relu2 = nn.LeakyReLU(0.2, inplace=True)
		self.att2 = SelfAttention(df_dim*4, (32, 32))
		
		self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(df_dim*8)
		self.relu3 = nn.LeakyReLU(0.2, inplace=True)
		self.att3 = SelfAttention(df_dim*8, (16, 16))

		self.conv4 = nn.Conv2d(df_dim*8, df_dim*8, 4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(df_dim*8)
		self.relu4 = nn.LeakyReLU(0.2, inplace=True)
		self.att4 = SelfAttention(df_dim*8, (8, 8))

		self.conv5 = nn.Conv2d(df_dim*8, df_dim*8, 4, 2, 1, bias=False)
		self.bn5 = nn.BatchNorm2d(df_dim*8)
		self.relu5 = nn.LeakyReLU(0.2, inplace=True)
		self.att5 = SelfAttention(df_dim*8, (4, 4))

		self.conv6 = nn.Conv2d(df_dim*8, df_dim*8, 4, 2, 1, bias=False)
		self.bn6 = nn.BatchNorm2d(df_dim*8)
		self.relu6 = nn.LeakyReLU(0.2, inplace=True)
		self.att6 = SelfAttention(df_dim*8, (2, 2))

		self.conv7 = nn.Conv2d(df_dim*8, df_dim*8, 4, 2, 1, bias=False)
		# self.bn7 = nn.BatchNorm2d(df_dim*8)
		self.relu7 = nn.LeakyReLU(0.2, inplace=True)

		self.conv8 = nn.Conv2d(df_dim*8, 1, 1, 1, 0, bias=False)
		self.sigmoid = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0.0, 0.02)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, input):
		h0 = self.att0(self.relu0(self.conv0(input)))
		h1 = self.att1(self.relu1(self.bn1(self.conv1(h0))))
		h2 = self.att2(self.relu2(self.bn2(self.conv2(h1))))
		h3 = self.att3(self.relu3(self.bn3(self.conv3(h2))))
		h4 = self.att4(self.relu4(self.bn4(self.conv4(h3))))
		h5 = self.att5(self.relu5(self.bn5(self.conv5(h4))))
		h6 = self.att6(self.relu6(self.bn6(self.conv6(h5))))
		h7 = self.relu7(self.conv7(h6))
		h8 = self.conv8(h7)
		output = self.sigmoid(h8)
		return h3, output.view(-1, 1).squeeze(1) # by squeeze, get just float not float Tenosor
		
		
if __name__ == "__main__":
	g = Generator(1,2,3)
	print(g.__class__.__name__)
