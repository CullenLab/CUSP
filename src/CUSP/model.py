import torch
from torch import nn

## network class

class InceptionBlock2D_Time(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes,dilation=1):
        super(InceptionBlock2D_Time,self).__init__()
        self.branches = nn.ModuleList()

        for kernel_size in kernel_sizes:
            padding = (kernel_size // 2)*dilation
            self.branches.append(
                nn.Sequential(nn.Conv2d(in_channels,out_channels,[1,kernel_size],padding=[0,padding],dilation=dilation),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU()
                              )
            )
        
    def forward(self,x):
        outputs = [conv(x) for conv in self.branches]
        x = torch.cat(outputs,dim=1)
        return x

class Up_InceptionBlock2D_Time(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(Up_InceptionBlock2D_Time,self).__init__()

        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            self.branches.append(
                nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,[1,kernel_size],padding=[0,padding]),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU()
                              )
            )
        
        
    def forward(self,x):
        
        xx = [conv(x) for conv in self.branches]
        x = torch.cat(xx,dim=1)

        return x

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttention2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.query = nn.Conv2d(in_channels, out_channels,[3,3],padding=[1,1])
        self.key = nn.Conv2d(in_channels, out_channels,[3,3],padding=[1,1])
        self.value = nn.Conv2d(in_channels, out_channels,[3,3],padding=[1,1])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: B*C*channel*time
        B,C,n_channel,n_time = x.shape[0],x.shape[1],x.shape[2],x.shape[3]

        query = self.query(x).view(B,self.out_channels,-1)
        key = self.key(x).view(B,self.out_channels,-1)
        value = self.value(x).view(B,self.out_channels,-1)
        
        # Compute attention scores
        # attention_scores = torch.bmm(query.transpose(1, 2), key)
        # attention_scores = self.softmax(attention_scores)
        
        # Apply attention scores to values
        attended_values = torch.bmm(self.softmax(torch.bmm(query.transpose(1, 2), key)), value.transpose(1, 2))
        attended_values = attended_values.transpose(1, 2)

        return attended_values.view(B,-1,n_channel,n_time)

class Down_SelfAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(Down_SelfAttention2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.query = nn.Conv2d(in_channels, out_channels, kernel,stride=kernel)
        self.key = nn.Conv2d(in_channels, out_channels, kernel,stride=kernel)
        self.value = nn.Conv2d(in_channels, out_channels, kernel,stride=kernel)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: B*C*channel*time
        B= x.shape[0]

        query = self.query(x)
        C,T = query.shape[2],query.shape[3]
        query = query.view(B,self.out_channels,-1)
        key = self.key(x).view(B,self.out_channels,-1)
        value = self.value(x).view(B,self.out_channels,-1)
        
        # Compute attention scores
        # attention_scores = torch.bmm(query.transpose(1, 2), key)
        # attention_scores = self.softmax(attention_scores)
        
        # Apply attention scores to values
        attended_values = torch.bmm(self.softmax(torch.bmm(query.transpose(1, 2), key)), value.transpose(1, 2))
        attended_values = attended_values.transpose(1, 2)

        return attended_values.view(B,self.out_channels,C,T)

class Up_SelfAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_SelfAttention2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.query = nn.Conv2d(in_channels, out_channels,[3,3],padding=[1,1])
        self.key = nn.Conv2d(in_channels, out_channels,[3,3],padding=[1,1])
        self.value = nn.Conv2d(in_channels, out_channels,[3,3],padding=[1,1])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_jump):
        # x: B*C*channel*time
        B,C,n_channel_up,n_time_up = x_jump.shape[0],x_jump.shape[1],x_jump.shape[2],x_jump.shape[3]

        query = self.query(x_jump).view(B,self.out_channels,-1)
        key = self.key(x).view(B,self.out_channels,-1)
        value = self.value(x).view(B,self.out_channels,-1)
        
        # Compute attention scores
        # attention_scores = torch.bmm(query.transpose(1, 2), key)
        # attention_scores = self.softmax(attention_scores)
        
        # Apply attention scores to values
        attended_values = torch.bmm(self.softmax(torch.bmm(query.transpose(1, 2), key)), value.transpose(1, 2))
        attended_values = attended_values.transpose(1, 2)

        return attended_values.view(B,self.out_channels,n_channel_up,n_time_up)

# CUSP full model
class Unet_CSDetect_SingleOutput(nn.Module):
    def __init__(self):
        super(Unet_CSDetect_SingleOutput, self).__init__()
        
        # input stage for ap
        self.ap_ib0 = InceptionBlock2D_Time(2,4,[3,7,11,15])
        self.ap_ib1 = InceptionBlock2D_Time(16,4,[3,7,11,15])
        self.ap_conv0 = nn.Conv2d(16,2,[1,1])
        # ap skip connection

        # input stage for lfp
        self.lfp_ib0 = InceptionBlock2D_Time(2,4,[3,7,11,15])
        self.lfp_ib1 = InceptionBlock2D_Time(16,4,[3,7,11,15])
        self.lfp_conv0 = nn.Conv2d(16,2,[1,1])
        # lfp skip connection

        # stack ap and lfp in dim=1, get B*2*C*T

        # combined down block
        # do time conv 
        self.down_ib0 = InceptionBlock2D_Time(4,4,[3,7,11,15])
        self.down_ib0_merge = nn.Conv2d(16,4,[1,1])
        self.down_ib0_bn = nn.BatchNorm2d(4)
        self.down_ib0_act = nn.ReLU()

        # do channel conv (use electrode local position info)
        self.down_chconv0 = nn.Conv2d(4,4,(4,1),stride=(4,1))
        self.down_bn0 = nn.BatchNorm2d(4)
        # time maxpool
        self.down_maxpool0 = nn.MaxPool2d((1,4),stride=(1,4),return_indices=True)

        # 2d cross-attention 
        self.down_attn0 = Down_SelfAttention2D(4,4,(4,4))
        # stack conv and attention and do [1,1] conv
        self.down_merge = nn.Conv2d(8,4,[1,1])
        self.down_bn1 = nn.BatchNorm2d(4)
        self.down_act0 = nn.ReLU()
        

        # combined basic block
        self.basic_ib0 = InceptionBlock2D_Time(4,4,[3,7,11,15])
        self.basic_ib0_merge = nn.Conv2d(16,4,[1,1])
        self.basic_ib0_bn = nn.BatchNorm2d(4)
        self.basic_ib0_act = nn.ReLU()

        self.basic_chconv0 = nn.Conv2d(4,4,(1,1),stride=(1,1))
        self.basic_bn0 = nn.BatchNorm2d(4)

        self.basic_attn0 = SelfAttention2D(4,4)

        self.basic_merge = nn.Conv2d(8,4,[1,1])
        self.basic_bn1 = nn.BatchNorm2d(4)
        self.basic_act0 = nn.ReLU()

        

        # combined up block
        self.up_ib0 = Up_InceptionBlock2D_Time(4,4,[3,7,11,15])
        self.up_ib0_merge = nn.Conv2d(16,4,[1,1])
        self.up_ib0_bn = nn.BatchNorm2d(4)
        self.up_ib0_act = nn.ReLU()

        self.up_chconv0 = nn.ConvTranspose2d(4,4,(4,1),stride=(4,1))
        self.up_bn0 = nn.BatchNorm2d(4)

        self.up_maxunpool0 = nn.MaxUnpool2d((1,4),stride=(1,4))
        
        self.up_attn0 = Up_SelfAttention2D(4,4)

        self.up_merge = nn.Conv2d(8,4,[1,1])
        self.up_bn1 = nn.BatchNorm2d(4)
        self.up_act0 = nn.ReLU()
        

        # output stage
        self.output_ib0 = InceptionBlock2D_Time(4,4,[3,7,11,15])
        self.output_ib0_merge = nn.Conv2d(16,4,[1,1])
        self.output_ib0_bn = nn.BatchNorm2d(4)
        self.output_ib0_act = nn.ReLU()
        self.output_ib1 = InceptionBlock2D_Time(4,4,[3,7,11,15])
        self.output_conv0 = nn.Conv2d(16,4,[1,1])
        self.output_bn0 = nn.BatchNorm2d(4)
        self.output_act0 = nn.ReLU()
        self.output_conv1 = nn.Conv2d(4,1,[1,1])
        self.output_bn1 = nn.BatchNorm2d(4)
        self.output_act1 = nn.ReLU()

        self.output_linear = nn.Linear(12,1)

    def htransforms(self, data):
        N = data.shape[-1]
        
        transforms = torch.fft.fft(data,dim=-1)
        transforms[:, :, :, 1:N//2]      *= -1j      # positive frequency
        transforms[:, :, :, (N+2)//2 + 1: N] *= +1j # negative frequency
        transforms[:, :, :, 0] = 0; # DC signal
        if N % 2 == 0:
            transforms[:, :, :, N//2] = 0; # the (-1)**n term
        
        # Do IFFT
        transforms = torch.fft.ifft(transforms,dim=-1)

        # return real and img part
        return torch.cat([torch.real(transforms),data],dim=1)

    def forward(self, lfp_input, ap_input):
        # lfp_input, ap_input: B*1*C*T

        # B,C,T = lfp_input.shape[0],lfp_input.shape[2],lfp_input.shape[3]

        # input stage for lfp
        lfp_input = self.htransforms(lfp_input)
        lfp = self.lfp_ib0(lfp_input)
        lfp = self.lfp_ib1(lfp)
        lfp = self.lfp_conv0(lfp)
        lfp += lfp_input # skip connection

        # input stage for ap
        ap_input = self.htransforms(ap_input)
        ap = self.ap_ib0(ap_input)
        ap = self.ap_ib1(ap)
        ap = self.ap_conv0(ap)
        ap += ap_input # skip connection

        # stack ap and lfp
        x = torch.cat([lfp,ap],dim=1)

        # combined down block
        x_copy = x.clone()
        x_Unet_skip = x.clone()
        x = self.down_ib0(x)
        x = self.down_ib0_merge(x)
        x = self.down_ib0_bn(x)
        x = self.down_ib0_act(x)

        down_pool0_input_size = x.size()
        x, down_pool0_indice = self.down_maxpool0(x)

        x = self.down_chconv0(x)
        x = self.down_bn0(x)

        x_attn = self.down_attn0(x_copy)

        x = torch.cat([x,x_attn],dim=1)
        x = self.down_merge(x)
        x = self.down_bn1(x)
        x = self.down_act0(x)


        # combined basic block
        x_copy = x.clone()
        x = self.basic_ib0(x)
        x = self.basic_ib0_merge(x)
        x = self.basic_ib0_bn(x)
        x = self.basic_ib0_act(x)

        x = self.basic_chconv0(x)
        x = self.basic_bn0(x)
        
        x_attn = self.basic_attn0(x_copy)

        x = torch.cat([x,x_attn],dim=1)
        x = self.basic_merge(x)
        x = self.basic_bn1(x)      
        x = self.basic_act0(x)


        # combined up block
        x_copy = x.clone()
        x = self.up_ib0(x)
        x = self.up_ib0_merge(x)
        x = self.up_ib0_bn(x)
        x = self.up_ib0_act(x)

        x = self.up_chconv0(x)
        x = self.up_bn0(x)
        x = self.up_maxunpool0(x, down_pool0_indice,output_size=down_pool0_input_size)

        x_attn = self.up_attn0(x_copy,x_Unet_skip)
        
        x = torch.cat([x,x_attn],dim=1)
        x = self.up_merge(x)
        x = self.up_bn1(x)  
        x = self.up_act0(x)
        
        

        # output stage
        x_skip = x.clone()
        x = self.output_ib0(x)
        x = self.output_ib0_merge(x)
        x = self.output_ib0_bn(x)
        x = self.output_ib0_act(x)
        x = self.output_ib1(x)
        x = self.output_conv0(x)
        x = self.output_bn0(x)
        x += x_skip
        x = self.output_act0(x)
        x = self.output_conv1(x)
        x = self.output_act1(x)

        x = x[:,0,:,:]
        x = self.output_linear(x.permute(0,2,1)).permute(0,2,1)

        # BCEWithLogitsLoss() has Sigmoid layer inside
        x = x[:,0,:]
        
        return x
