import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp


class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class GroupLinearLayer(nn.Module):

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w     = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)


class GroupLSTMCell(nn.Module):
    """
    GroupLSTMCell can compute the operation of N LSTM Cells at once.
    """
    def __init__(self, inp_size, hidden_size, num_lstms):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        
        self.i2h = GroupLinearLayer(inp_size, 4 * hidden_size, num_lstms)
        self.h2h = GroupLinearLayer(hidden_size, 4 * hidden_size, num_lstms)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hid_state):
        """
        input: x (batch_size, num_lstms, input_size)
               hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
        output: h (batch_size, num_lstms, hidden_state)
                c ((batch_size, num_lstms, hidden_state))
        """
        h, c = hid_state
        preact = self.i2h(x) + self.h2h(h)

        gates = preact[:, :,  :3 * self.hidden_size].sigmoid()
        g_t = preact[:, :,  3 * self.hidden_size:].tanh()
        i_t = gates[:, :,  :self.hidden_size]
        f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, :, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t) 
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t


class GroupGRUCell(nn.Module):
    """
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """
    def __init__(self, input_size, hidden_size, num_grus):
        super(GroupGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = GroupLinearLayer(input_size, 3 * hidden_size, num_grus)
        self.h2h = GroupLinearLayer(hidden_size, 3 * hidden_size, num_grus)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data = torch.ones(w.data.size())#.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """
        input: x (batch_size, num_grus, input_size)
               hidden (batch_size, num_grus, hidden_size)
        output: hidden (batch_size, num_grus, hidden_size)
        """
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
        
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy



class RIMCell_global_working_space(nn.Module):
    '''
    添加global working space
    '''
    def __init__(self, 
        device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size = 64, input_value_size = 400, input_query_size = 64,
        num_input_heads = 1, input_dropout = 0.1, num_mem_slots=4, write_attention_heads=1, key_size=32,
              mem_attention_heads=4, mem_attention_key=32, write_dropout=0.1, read_dropout=0.1, num_mlp_layers=1):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_units =num_units
        self.rnn_cell = rnn_cell
        self.key_size = input_key_size
        self.k = k
        self.num_mem_slots = num_mem_slots
        self.num_input_heads = num_input_heads
        self.input_key_size = input_key_size
        self.input_query_size = input_query_size
        self.input_value_size = input_value_size

        self.key = nn.Linear(input_size, num_input_heads * input_query_size).to(self.device)
        self.value = nn.Linear(input_size, num_input_heads * input_value_size).to(self.device)

        if self.rnn_cell == 'GRU':
            self.rnn = GroupGRUCell(input_value_size, hidden_size, num_units)
            self.query = GroupLinearLayer(hidden_size,  input_key_size * num_input_heads, self.num_units)
        else:
            self.rnn = GroupLSTMCell(input_value_size, hidden_size, num_units)
            self.query = GroupLinearLayer(hidden_size,  input_key_size * num_input_heads, self.num_units)
        self.input_dropout = nn.Dropout(p =input_dropout)

        self.global_working_space = Global_working_space(self.hidden_size, self.hidden_size, self.num_mem_slots, write_attention_heads, key_size,
              mem_attention_heads, mem_attention_key, write_dropout, read_dropout, num_mlp_layers)


    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def input_attention_mask(self, x, h):
        """
        Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
                h (batch_size, num_units, hidden_size)
        Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
                mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
        """
        key_layer = self.key(x)  # (batch_size, 2, num_input_heads * input_query_size)
        value_layer = self.value(x)  # (batch_size, 2, num_input_heads * input_value_size)
        query_layer = self.query(h)  # (batch_size, num_units, num_input_heads * input_key_size)
        # 注意这不是self-attention了，而是模块的h到输入的x的attention
        key_layer = self.transpose_for_scores(key_layer,  self.num_input_heads, self.input_key_size)  # (batch_size, num_input_heads, 2, input_key_size)
        value_layer = torch.mean(self.transpose_for_scores(value_layer,  self.num_input_heads, self.input_value_size), dim = 1)  # (batch_size, 2, input_value_size)
        query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)  # (batch_size, num_input_heads, num_units, input_query_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size)  # (batch_size, num_input_heads, num_units, 2)
        attention_scores = torch.mean(attention_scores, dim = 1)  # (batch_size, num_units, 2)
        mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)  # (batch_size, num_units)

        not_null_scores = attention_scores[:,:, 0]  # (batch_size, num_units)选择attention在有意义输入（而非null）
        topk1 = torch.topk(not_null_scores, self.k, dim = 1)  # ((batch_size, self.k),(batch_size, self.k))
        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, self.k)  # (batch_size*self.k)

        mask_[row_index, topk1.indices.view(-1)] = 1  # 这里row_index和topk1.indices.view代表的column index结合指示出了mask_中为1的坐标！！！
        
        attention_probs = self.input_dropout(nn.Softmax(dim = -1)(attention_scores))  # (batch_size, num_units, 2)
        inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)  # (batch_size, num_units, input_value_size), (batch_size, num_units)

        return inputs, mask_


    def forward(self, x, mem, hs, cs = None):
        """
        Input : x (batch_size, 1 , input_size)
                mem (batch_size, num_mem_slots, num_mem_size)
                hs (batch_size, num_units, hidden_size)
                cs (batch_size, num_units, hidden_size)
        Output: new hs, cs for LSTM
                new hs for GRU
        """
        size = x.size()
        null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
        x = torch.cat((x, null_input), dim = 1)

        # Compute input attention
        inputs, mask = self.input_attention_mask(x, hs)  # (batch_size, num_units, input_value_size), (batch_size, num_units)
        h_old = hs * 1.0
        if cs is not None:
            c_old = cs * 1.0
        

        # Compute RNN(LSTM or GRU) output
        
        if cs is not None:
            hs, cs = self.rnn(inputs, (hs, cs))
        else:
            hs = self.rnn(inputs, hs)

        # Block gradient through inactive units
        mask = mask.unsqueeze(2)  # (batch_size, num_units, 1)
        h_new = blocked_grad.apply(hs, mask)  # 定义了一个不改变前向传播，只mask反向梯度的torch算子

        hs = mask * h_new + (1 - mask) * h_old  # 非激活模块不更新状态
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old

        # Compute communication attention
        hs_, mem_new = self.global_working_space(hs[mask.squeeze(2).bool()].view(-1,self.k,self.hidden_size),mem)
        # TODO：为了让选择可导，只能出此下策
        # hs[mask.squeeze(2).bool()] = hs[mask.squeeze(2).bool()]-hs[mask.squeeze(2).bool()].data.clone().detach()+hs_.view(-1, self.hidden_size)
        hs = hs.clone().detach()
        hs[mask.squeeze(2).bool()] = hs_.view(-1, self.hidden_size)
        # mem_new = self.global_working_space.write(hs[mask.squeeze(2).bool()].view(-1,self.k,self.hidden_size), mem)
        # hs = self.global_working_space.read(hs, mem_new)

        return hs, mem_new, cs


class Global_working_space(nn.Module):
    def __init__(self, write_input_size, num_mem_size, num_mem_slots=4, write_attention_heads=1, key_size=32,
              mem_attention_heads=4, mem_attention_key=32, write_dropout=0.1, read_dropout=0.1, num_mlp_layers=1):
        '''
        write_input_size:读写working space的h的维度
        num_mem_size:working space每一个slot的维度
        num_mem_slots:working space的slot数目
        write_attention_head:写multi-head attention的head数目
        '''
        super().__init__()
        self.write_input_size = write_input_size  # 要写入存储空间的输入，一般是RIM各个unit的hidden
        self.num_mem_slots = num_mem_slots
        self.num_mem_size = num_mem_size
        self.write_attention_heads = write_attention_heads
        self.key_size = key_size
        self.num_mlp_layers = num_mlp_layers
        assert self.write_input_size==self.num_mem_size, "目前的模型为了之后计算attention时concat，规定这二者要相等"

        self.write_query = nn.Linear(
            self.num_mem_size, self.write_attention_heads*self.key_size)
        # TODO:此处对于各个memory cell是否需要采用GroupLinear进行特异化的attention key计算?
        self.write_key = nn.Linear(
            self.write_input_size, self.write_attention_heads*self.key_size)
        self.write_value = nn.Linear(self.write_input_size, self.write_attention_heads*self.num_mem_size)
        self.write_output = nn.Linear(self.num_mem_size*self.write_attention_heads, self.num_mem_size)
        self.write_dropout_layer = nn.Dropout(p=write_dropout)
        self.write_attention_mlp = nn.Sequential(
            *[nn.Linear(self.num_mem_size, self.num_mem_size) for _ in range(self.num_mlp_layers)],
            nn.LayerNorm((self.num_mem_slots, self.num_mem_size, self.num_mem_size)),
            nn.ReLU()
        )

        # 这里直接默认gating_style是units，即对每个memory slot生成单独的gate，如果style是memory的话，输出维度是1，应用到整个memory上
        self.update_trans = nn.Linear(self.write_input_size, self.num_mem_size)
        self.update_gating = nn.Linear(self.num_mem_size, self.num_mem_size*2)

        self.mem_attention_heads = mem_attention_heads
        self.mem_attention_key = mem_attention_key
        self.mem_attention_value = write_input_size
        self.read_dropout = read_dropout

        self.read_query = nn.Linear(
            self.write_input_size, self.mem_attention_heads*self.mem_attention_key)
        self.read_key = nn.Linear(
            self.num_mem_size, self.mem_attention_heads*self.mem_attention_key)
        self.read_value = nn.Linear(self.num_mem_size, self.mem_attention_heads*self.write_input_size)
        self.read_output = nn.Linear(self.mem_attention_value*self.mem_attention_heads, self.mem_attention_value)
        self.read_dropout_layer = nn.Dropout(p=read_dropout)

    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def write(self, write_inputs, memory):
        '''
        write_inputs:[batch_size, num_units?, hidden_size]
        memory:[batch_size, num_mem_slots, num_mem_size]
        '''
        write_inputs = torch.cat([write_inputs, memory],dim=1)  # 理论上inputs和memory的最后一维形状未必一样，但是文章中这么规定了。。。。。。
        query_write = self.write_query(memory)  # [batch_size, num_mem_slots, size_attention_head*key_size]
        key_write = self.write_key(write_inputs) # [batch_size, num_units?+num_mem_slots, size_attention_head*key_size]
        value_write = self.write_value(write_inputs) # [batch_size, num_units?+num_mem_slots, size_attention_head*num_mem_size]

        query_write = self.transpose_for_scores(query_write, self.write_attention_heads, self.key_size)  # (batch_size, size_attention_head, num_mem_slots, key_size)
        key_write = self.transpose_for_scores(key_write, self.write_attention_heads, self.key_size)  # (batch_size, size_attention_head, num_units, key_size)
        value_write = self.transpose_for_scores(value_write, self.write_attention_heads, self.num_mem_size)  # (batch_size, size_attention_head, num_units, num_mem_size)
        attention_scores = torch.matmul(query_write, key_write.transpose(-1, -2))  # (batch_size, size_attention_head, num_mem_slots, num_units)
        attention_scores = attention_scores / math.sqrt(self.key_size)

        attention_probs = self.write_dropout_layer(nn.Softmax(dim=-1)(attention_scores))  # (batch_size, size_attention_head, num_mem_slots, num_units)

        memory_new = torch.matmul(attention_probs, value_write)  # (batch_size, size_attention_head, num_mem_slots, num_mem_size)
        memory_new = self.write_output(memory_new.transpose(-2,-3).view((-1,self.num_mem_slots,self.write_attention_heads*self.num_mem_size)))

        return memory_new

    def update(self, gating_inputs, gating_memory, new_memory):
        '''
        gating_inputs: [B, Ns, hidden_size]
        gating_memory: [B, num_mem_slots, num_mem_size]
        '''
        inputs_mean = torch.relu(self.update_trans(gating_inputs)).mean(dim=1,keepdim=True)  # [B, 1, num_mem_size]
        k = inputs_mean+torch.tanh(gating_memory)
        gates = torch.sigmoid(self.update_gating(k))
        input_gate, forget_gate = torch.split(gates, self.num_mem_size, dim=-1)
        new_memory = input_gate*torch.tanh(new_memory)+forget_gate*gating_memory
        return new_memory

    def read(self, read_heads, memory):
        query_read = self.read_query(read_heads)
        key_read = self.read_key(memory)
        value_read = self.read_value(memory)

        query_read = self.transpose_for_scores(query_read, self.mem_attention_heads, self.mem_attention_key)  # (batch_size, mem_attention_heads, num_units?, mem_attention_key)
        key_read = self.transpose_for_scores(key_read, self.mem_attention_heads, self.mem_attention_key)  # (batch_size, mem_attention_heads, num_mem_slots, mem_attention_key)
        value_read = self.transpose_for_scores(value_read, self.mem_attention_heads, self.mem_attention_value)
        attention_scores = torch.matmul(query_read, key_read.transpose(-1, -2))  # (batch_size, mem_attention_heads, num_units?, num_mem_slots)
        attention_scores = attention_scores / math.sqrt(self.mem_attention_key)

        attention_probs = self.read_dropout_layer(nn.Softmax(dim=-1)(attention_scores))  # (batch_size, mem_attention_heads, num_units?, num_mem_slots)

        read_heads_new = torch.matmul(attention_probs, value_read)  # (batch_size, mem_attention_heads, num_units?, write_input_size)
        num_units=read_heads_new.size(2)
        read_heads_new = self.read_output(read_heads_new.transpose(-2,-3).reshape(-1,num_units,self.mem_attention_heads*self.write_input_size))  # (batch_size, num_units, write_input_size)

        return read_heads+read_heads_new

    def forward(self, h_s, mem):
        mem_new = self.write(h_s, mem)
        mem_new = self.update(h_s, mem, mem_new)
        h_s = self.read(h_s, mem_new)
        return h_s, mem_new


class GRIM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_units, k, num_mem_slots, rnn_cell, n_layers, bidirectional, **kwargs):
        super().__init__()
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1
        self.rnn_cell = rnn_cell
        self.num_units = num_units
        self.hidden_size = hidden_size
        if self.num_directions == 2:
            self.rimcell = nn.ModuleList([RIMCell_global_working_space(self.device, input_size, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) if i < 2 else 
                RIMCell_global_working_space(self.device, 2 * hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) for i in range(self.n_layers * self.num_directions)])
        else:
            self.rimcell = nn.ModuleList([RIMCell_global_working_space(self.device, input_size, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) if i == 0 else
            RIMCell_global_working_space(self.device, hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) for i in range(self.n_layers)])

    def layer(self, rim_layer, x, h, c = None, direction = 0):
        batch_size = x.size(1)
        xs = list(torch.split(x, 1, dim = 0))
        if direction == 1: xs.reverse()
        hs = h.squeeze(0).view(batch_size, self.num_units, -1)
        cs = None
        if c is not None:
            cs = c.squeeze(0).view(batch_size, self.num_units, -1)
        outputs = []
        for x in xs:
            x = x.squeeze(0)
            hs, cs = rim_layer(x.unsqueeze(1), hs, cs)
            outputs.append(hs.view(1, batch_size, -1))
        if direction == 1: outputs.reverse()
        outputs = torch.cat(outputs, dim = 0)
        if c is not None:
            return outputs, hs.view(batch_size, -1), cs.view(batch_size, -1)
        else:
            return outputs, hs.view(batch_size, -1)

    def forward(self, x, h = None, c = None):
        """
        Input: x (seq_len, batch_size, feature_size
               h (num_layers * num_directions, batch_size, hidden_size * num_units)
               c (num_layers * num_directions, batch_size, hidden_size * num_units)
        Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
                h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
        """

        hs = torch.split(h, 1, 0) if h is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
        hs = list(hs)
        cs = None
        if self.rnn_cell == 'LSTM':
            cs = torch.split(c, 1, 0) if c is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
            cs = list(cs)
        for n in range(self.n_layers):
            idx = n * self.num_directions
            if cs is not None:
                x_fw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx])
            else:
                x_fw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None)
            if self.num_directions == 2:
                idx = n * self.num_directions + 1
                if cs is not None:
                    x_bw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx], direction = 1)
                else:
                    x_bw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None, direction = 1)

                x = torch.cat((x_fw, x_bw), dim = 2)
            else:
                x = x_fw
        hs = torch.stack(hs, dim = 0)
        if cs is not None:
            cs = torch.stack(cs, dim = 0)
            return x, hs, cs
        return x, hs


if __name__=="__main__":
    rim_cell = RIMCell_global_working_space("cuda", 10, 20, 6, 4, "LSTM").cuda()
    a = torch.randn(8,14*14,10).cuda()
    mem = torch.zeros((8,4,20)).cuda()
    h = torch.zeros((8,6,20)).cuda()
    c = torch.zeros((8,6,20)).cuda()
    h_new, mem_new, c_new = rim_cell(a[:,1:1,:],mem,h,c)