# %% [markdown]
# # Attribute
# 
# **Original Work**: *Maziar Raissi, Paris Perdikaris, and George Em Karniadakis*
# 
# **Github Repo** : https://github.com/maziarraissi/PINNs
# 
# **Link:** https://github.com/maziarraissi/PINNs/tree/master/appendix/continuous_time_identification%20(Burgers)
# 
# @article{raissi2017physicsI,
#   title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
#   author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
#   journal={arXiv preprint arXiv:1711.10561},
#   year={2017}
# }
# 
# @article{raissi2017physicsII,
#   title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
#   author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
#   journal={arXiv preprint arXiv:1711.10566},
#   year={2017}
# }

# %% [markdown]
# ## Libraries and Dependencies

# %%
import sys, os
filepath = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(filepath))
utilities_dir = os.path.join(root_dir, "Utilities")
data_path = os.path.join(root_dir, 'Burgers Equation', 'data', 'burgers_shock.mat')
sys.path.insert(0, utilities_dir)
print (utilities_dir)

import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

np.random.seed(1234)
LBFGS_iterations = 50000
Adam_iterations = 10000

# %%
# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %% [markdown]
# ## Physics-informed Neural Networks

# %%
# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out

# %%
# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, x0, u0, x1, u1, layers, dt, lb, ub, q):
        
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        
        # input x data
        self.x0 = torch.tensor(x0, requires_grad=True).float().to(device)
        self.x1 = torch.tensor(x1, requires_grad=True).float().to(device)

        # training gt data
        self.u0 = torch.tensor(u0).float().to(device)
        self.u1 = torch.tensor(u1).float().to(device)

        # settings
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(device)
        
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        
        # deep neural networks
        self.layers = layers
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)
        
        self.dt = torch.tensor(dt).float().to(device)
        self.q = max(q,1)
        
        # dummy variables for gradient calculation
        self.dummy_x0 = torch.tensor(np.ones((self.x0.shape[0], self.q)), requires_grad=True).float().to(device)
        self.dummy_x1 = torch.tensor(np.ones((self.x1.shape[0], self.q)), requires_grad=True).float().to(device)

        # Load IRK weights
        weight_file = os.path.join(utilities_dir, 'IRK_weights', 'Butcher_IRK%d.txt' % (q))
        tmp = np.float32(np.loadtxt(weight_file, ndmin = 2))
        weights =  np.reshape(tmp[0:q**2+q], (q+1,q))     
        self.IRK_alpha = torch.tensor(weights[0:-1,:]).float().to(device)
        self.IRK_beta = torch.tensor(weights[-1:,:]).float().to(device)        
        self.IRK_times = torch.tensor(tmp[q**2+q:]).float().to(device)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=LBFGS_iterations, 
            max_eval=LBFGS_iterations, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
        
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def fwd_gradients_0(self, u, x):        
        g = torch.autograd.grad(
            u, x, 
            grad_outputs=self.dummy_x0,
            retain_graph=True,
            create_graph=True
        )[0]
        return torch.autograd.grad(
            g, self.dummy_x0, 
            grad_outputs=torch.ones_like(g),
            retain_graph=True,
            create_graph=True
        )[0]

    def fwd_gradients_1(self, u, x):        
        g = torch.autograd.grad(
            u, x, 
            grad_outputs=self.dummy_x1,
            retain_graph=True,
            create_graph=True
        )[0]
        return torch.autograd.grad(
            g, self.dummy_x1, 
            grad_outputs=torch.ones_like(g),
            retain_graph=True,
            create_graph=True
        )[0]

    def net_u0(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        u = self.dnn(x)        
        u_x = self.fwd_gradients_0(u, x)
        u_xx = self.fwd_gradients_0(u_x, x)
        f = -lambda_1*u*u_x + lambda_2*u_xx
        u0 = u - self.dt*torch.matmul(f, self.IRK_alpha.T)
        return u0
    
    def net_u1(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        u = self.dnn(x)        
        u_x = self.fwd_gradients_1(u, x)
        u_xx = self.fwd_gradients_1(u_x, x)
        f = -lambda_1*u*u_x + lambda_2*u_xx
        u1 = u + self.dt*torch.matmul(f, (self.IRK_beta - self.IRK_alpha).T)
        return u1

    def loss_func(self):
        u0_pred = self.net_u0(self.x0)
        u1_pred = self.net_u1(self.x1)
        loss = torch.sum((self.u0 - u0_pred) ** 2) + torch.sum((self.u1 - u1_pred) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Loss: %e, l1: %.5f, l2: %.5f' % 
                (
                    loss.item(), 
                    self.lambda_1.item(), 
                    torch.exp(self.lambda_2.detach()).item()
                )
            )
        return loss
    
    def train(self, nIter):
        self.dnn.train()

        # dummy variables for gradient calculation
        self.dummy_x0 = torch.tensor(np.ones((self.x0.shape[0], self.q)), requires_grad=True).float().to(device)
        self.dummy_x1 = torch.tensor(np.ones((self.x1.shape[0], self.q)), requires_grad=True).float().to(device)
        
        for epoch in range(nIter):
            u0_pred = self.net_u0(self.x0)
            u1_pred = self.net_u1(self.x1)
            loss = torch.sum((self.u0 - u0_pred) ** 2) + torch.sum((self.u1 - u1_pred) ** 2)
            
            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            
            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' % 
                    (
                        epoch, 
                        loss.item(), 
                        self.lambda_1.item(), 
                        torch.exp(self.lambda_2).item()
                    )
                )
                
        # Backward and optimize
        self.optimizer.step(self.loss_func)
    
    def predict(self, x_star):
        x_star = torch.tensor(x_star, requires_grad=True).float().to(device)

        # dummy variables for gradient calculation
        self.dummy_x0 = torch.tensor(np.ones((x_star.shape[0], self.q)), requires_grad=True).float().to(device)
        self.dummy_x1 = torch.tensor(np.ones((x_star.shape[0], self.q)), requires_grad=True).float().to(device)

        self.dnn.eval()
        u0_star = self.net_u0(x_star)
        u1_star = self.net_u1(x_star)

        return u0_star, u1_star

# %% [markdown]
# ## Configurations

# %%
skip = 80

N0 = 199
N1 = 201

data = scipy.io.loadmat(data_path)

t_star = data['t'].flatten()[:,None]
x_star = data['x'].flatten()[:,None]
Exact = np.real(data['usol'])

idx_t = 10

######################################################################
######################## Noiseles Data ###############################
######################################################################
noise = 0.0    
    
idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
x0 = x_star[idx_x,:]
u0 = Exact[idx_x,idx_t][:,None]
u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    
idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
x1 = x_star[idx_x,:]
u1 = Exact[idx_x,idx_t + skip][:,None]
u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])

dt = t_star[idx_t+skip] - t_star[idx_t]   
print('dt', dt)     
q = int(np.ceil(0.5*np.log(np.finfo(float).eps)/np.log(dt)))

layers = [1, 50, 50, 50, 50, q]


# Doman bounds
lb = x_star.min(0)
ub = x_star.max(0)

model = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)
model.train(Adam_iterations)

U0_pred, U1_pred = model.predict(x_star)

lambda_1_value = model.lambda_1.detach().cpu().numpy()      # different from TF
lambda_2_value = model.lambda_2.detach().cpu().numpy()      # different from TF
lambda_2_value = np.exp(lambda_2_value)

nu = 0.01/np.pi       
error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

######################################################################
############################# Plotting ###############################
######################################################################

fig, ax = newfig(1.0, 1.5)
ax.axis('off')

gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3+0.05, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])
    
h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow',
                extent=[t_star.min(),t_star.max(), lb[0], ub[0]],
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x_star.min(), x_star.max(), 2)[:,None]
ax.plot(t_star[idx_t]*np.ones((2,1)), line, 'w-', linewidth = 1.0)
ax.plot(t_star[idx_t + skip]*np.ones((2,1)), line, 'w-', linewidth = 1.0)    
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$u(t,x)$', fontsize = 10)

plt.show()

gs1 = gridspec.GridSpec(1, 2)
gs1.update(top=1-1/3-0.1, bottom=1-2/3, left=0.15, right=0.85, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x_star,Exact[:,idx_t][:,None], 'b', linewidth = 2, label = 'Exact')
ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t], u0.shape[0]), fontsize = 10)

ax = plt.subplot(gs1[0, 1])
ax.plot(x_star,Exact[:,idx_t + skip][:,None], 'b', linewidth = 2, label = 'Exact')
ax.plot(x1, u1, 'rx', linewidth = 2, label = 'Data')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t+skip], u1.shape[0]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

plt.show()
