
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

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# ## Physics-informed Neural Networks

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

# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, x0, u0, x1, layers, dt, lb, ub, q):
        
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        
        # input x data
        self.x0 = torch.tensor(x0, requires_grad=True).float().to(device)
        self.x1 = torch.tensor(x1, requires_grad=True).float().to(device)

        # training gt data
        self.u0 = torch.tensor(u0).float().to(device)
        
        # deep neural networks
        self.layers = layers
        self.dnn = DNN(layers).to(device)
        
        self.dt = torch.tensor(dt).float().to(device)
        self.q = max(q,1)
        
        # dummy variables for gradient calculation
        self.dummy_x0 = torch.tensor(np.ones((self.x0.shape[0], self.q)), requires_grad=True).float().to(device)
        self.dummy_x1 = torch.tensor(np.ones((self.x1.shape[0], self.q+1)), requires_grad=True).float().to(device)

        # Load IRK weights
        weight_file = os.path.join(utilities_dir, 'IRK_weights', 'Butcher_IRK%d.txt' % (q))
        tmp = np.float32(np.loadtxt(weight_file, ndmin = 2))
        self.IRK_weights =  torch.tensor(np.reshape(tmp[0:q**2+q], (q+1,q))).float().to(device)           
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
        nu = 0.01/np.pi
        u1 = self.dnn(x)  
        u = u1[:,:-1]      
        u_x = self.fwd_gradients_0(u, x)
        u_xx = self.fwd_gradients_0(u_x, x)
        f = -u*u_x + nu*u_xx
        u0 = u1 - self.dt*torch.matmul(f, self.IRK_weights.T)
        return u0
    
    def net_u1(self, x):
        u1 = self.dnn(x)
        return u1

    def loss_func(self):
        u0_pred = self.net_u0(self.x0)
        u1_pred = self.net_u1(self.x1)
        loss = torch.sum((self.u0 - u0_pred) ** 2) + torch.sum(u1_pred ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        
        self.iter += 1
        if self.iter % 100 == 0:
            print('Loss: %e' % loss.item())
        return loss
    
    def train(self, nIter):
        self.dnn.train()

        for epoch in range(nIter):
            u0_pred = self.net_u0(self.x0)
            u1_pred = self.net_u1(self.x1)
            loss = torch.sum((self.u0 - u0_pred) ** 2) + torch.sum(u1_pred ** 2)
            
            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            
            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e' % 
                    (
                        epoch, 
                        loss.item()
                    )
                )
                
        # Backward and optimize
        self.optimizer.step(self.loss_func)
    
    def predict(self, x_star):
        x_star = torch.tensor(x_star, requires_grad=True).float().to(device)

        self.dnn.eval()
        u1_star = self.net_u1(x_star)

        return u1_star.detach().cpu().numpy()

# ## Configurations
        
q = 500
layers = [1, 50, 50, 50, q+1]
lb = np.array([-1.0])
ub = np.array([1.0])

N = 250

data = scipy.io.loadmat(data_path)

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

idx_t0 = 10
idx_t1 = 90
dt = t[idx_t1] - t[idx_t0]

# Initial data
noise_u0 = 0.0
idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
x0 = x[idx_x,:]
u0 = Exact[idx_t0:idx_t0+1,idx_x].T
u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])

# Boudanry data
x1 = np.vstack((lb,ub))

# Test data
x_star = x

model = PhysicsInformedNN(x0, u0, x1, layers, dt, lb, ub, q)
model.train(Adam_iterations)

U1_pred = model.predict(x_star)

error = np.linalg.norm(U1_pred[:,-1] - Exact[idx_t1,:], 2)/np.linalg.norm(Exact[idx_t1,:], 2)
print('Error: %e' % (error))


######################################################################
############################# Plotting ###############################
######################################################################    

fig, ax = newfig(1.0, 1.2)
ax.axis('off')

####### Row 0: h(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x_star.min(), x_star.max()], 
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
    
line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[idx_t0]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[idx_t1]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
leg = ax.legend(frameon=False, loc = 'best')
ax.set_title('$u(t,x)$', fontsize = 10)
    
plt.show()
    
####### Row 1: h(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 2)
gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[idx_t0,:], 'b-', linewidth = 2) 
ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')      
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
ax.set_xlim([lb-0.1, ub+0.1])
ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)


ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact[idx_t1,:], 'b-', linewidth = 2, label = 'Exact') 
ax.plot(x_star, U1_pred[:,-1], 'r--', linewidth = 2, label = 'Prediction')      
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize = 10)    
ax.set_xlim([lb-0.1, ub+0.1])

ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)

plt.show()
