# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:55:11 2019

@author: Administrator
"""

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, digamma
from math import log
import torch.optim as optim 

z_true = 3.0
a0, b0 = 1.0, 1.0

class Log_gamma(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input.detach().numpy()
        out = np.array(gammaln(input_np))   # 莫名其妙这里就要加上np.array
        out = torch.from_numpy(out).to(device=input.device).type(dtype=input.dtype)
        
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.digamma(input) * grad_output
        
        return grad_input

log_gamma = Log_gamma.apply
    
def unwrap(theta):
    alpha = torch.exp(theta[0]) + 1
    beta = torch.exp(theta[1])
    return alpha, beta
    
def gamma_entropy(theta):
    alpha, beta = unwrap(theta)
    return alpha - torch.log(beta) + log_gamma(alpha) + (1-alpha)*torch.digamma(alpha)
    
def log_p(x, z):     # 每一批x和一个z中的数值对应一个输出，因此return的形状和z的形状是一样的
    log_px_z = torch.sum(x*torch.log(z[:, None]) - z[:, None] - log_gamma(x+1), 1) 
    log_pz = a0*log(b0) - np.float(gammaln(a0)) + (a0-1)*torch.log(z)  - b0*z    # gammaln必须加上np.float这操作是个啥
    return log_px_z + log_pz
  
def h(e, a):
    return (a- 1/3)*(1 + e/torch.sqrt(9*a-3))**3

def h_inv(z,a):
    return torch.sqrt(9*a-3)*((z/(a-1/3))**(1/3)-1)

def dhde(e,a):
    return (a-1/3)*3/torch.sqrt(9*a - 3)*(1+e/torch.sqrt(9*a - 3))**2

def log_q(z, a):
    return (a-1)*torch.log(z) - z - log_gamma(a)

class elbo(torch.autograd.Function): #因为在这个class里面没有办法使用autograd，因此只能外部计算好以后，传入到这里
    @staticmethod
    def forward(ctx, alpha, beta, x, z, g_alpha, g_beta):
        ctx.g_beta = g_beta
        ctx.g_alpha = g_alpha
        return torch.mean(log_p(x, z/beta))
    @staticmethod
    def backward(ctx, grad_output):
        return ctx.g_alpha*grad_output, ctx.g_beta*grad_output, None,None,None,None
    
El = elbo.apply

if __name__ == '__main__':
    a_list = []
    loss_list = []
    
    N = 20
    x = torch.FloatTensor(np.random.poisson(z_true, size = N))
    alpha_true = a0 + x.sum().cpu().numpy()
    beta_true = b0 + N
  #  z = torch.FloatTensor(np.random.gamma(a0,1, size = 3))
    theta = Variable(torch.FloatTensor([1,1]), requires_grad = True)
 #   alpha, beta = unwrap(theta) 
    optimizer = optim.Adam([theta], lr=1e-1)
   # alpha = Variable(torch.FloatTensor([np.exp(1) + 1]), requires_grad =True)
   # beta = Variable(torch.FloatTensor([np.exp(1)]), requires_grad = True)
    for i in range(10000):
        alpha, beta = unwrap(theta)
        a_list.append(alpha.data)
        optimizer.zero_grad()
        z = torch.FloatTensor(np.random.gamma(alpha.detach().numpy(), 1, size = 10))  # variable类型数据变成numpy需要用到detach
     #   p = Variable(torch.FloatTensor([1]), requires_grad = True)
        
        g_beta = torch.autograd.grad(outputs = log_p(x, z/beta), inputs = beta, grad_outputs=torch.ones(z.shape))
        g_beta = g_beta[0]/z.shape[0]
        
        vap = h_inv(z,alpha).data # 只取数值
        g_alpha_1 = torch.autograd.grad(outputs = log_p(x, h(vap, alpha)/beta.data), inputs = alpha, grad_outputs=torch.ones(vap.shape))
        g_alpha_1 = g_alpha_1[0]/vap.shape[0]
      
        fun = log_p(x, h(vap, alpha)/beta.data).data*(log_q(h(vap, alpha), alpha) + torch.log(dhde(vap, alpha)))  # 使用.data可以只用数值
        g_alpha_2 = torch.autograd.grad(outputs = fun, inputs = alpha, grad_outputs=torch.ones(vap.shape))
        g_alpha_2 = g_alpha_2[0]/vap.shape[0]
    
        g_alpha = g_alpha_1 + g_alpha_2
    
        loss = - gamma_entropy(theta) - El(alpha, beta, x,z,g_alpha,g_beta)
        loss_list.append(loss.item())
        
        loss.backward(retain_graph=True)
        optimizer.step()
        
        if (i+1)%100==0:
           print('loss: %.4f' %loss.item())
           print('alpha:%.4f -- beta:%.4f --' %(alpha.detach().numpy(), beta.detach().numpy()))
         
   # print('alpha:%.4f -- beta:%.4f --' %(alpha.detach().numpy(), beta.detach().numpy()))
    a = np.array(a_list)       
    plt.plot(a)
    print(np.mean(a[2000:]))
    print(np.std(a[2000:]))
    # 下一步思路是反向截断梯度，也就是说在回传梯度的时候，看看有没有办法只计算到中间某个变量
        
    
    
    
    
    
    
    

    



    
        
    
    
    

    
