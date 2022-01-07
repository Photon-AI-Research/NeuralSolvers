import numpy as np
import torch
from torch import Tensor, ones, stack, load
from pyDOE import lhs
import matplotlib.pyplot as plt

class Sampler:
    def __init__(self, geometry, num_points, sampler="lhs"):
        self.geometry = geometry
        self.num_points = num_points
        self.sampler = sampler
   
    def sample(self, model, pde):

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


      if self.sampler == "lhs":
        np.random.seed(42)
        lb = self.geometry.xf[0]
        ub = self.geometry.xf[1]

        xf = lb + (ub - lb) * lhs(2, self.num_points)
        weight = np.ones_like(xf, shape =self.num_points) 

        self.xf= torch.tensor(xf).float()  
        self.weight = torch.tensor(weight).float()
        
      if self.sampler == "adaptive":
        np.random.seed(42)
        ns = int(self.num_points/8)
        lb = self.geometry.xf[0]
        ub = self.geometry.xf[1]

        # random seeds
        random_x = np.random.uniform(lb[0], ub[0], ns).reshape(-1, 1)
        random_t = np.random.uniform(lb[1], ub[1], ns).reshape(-1, 1)

        #random_x = np.linspace(lb[0], ub[0], int(np.sqrt(ns)))
        #random_t = np.linspace(lb[1], ub[1], int(np.sqrt(ns)))

        #x_grid, t_grid = np.meshgrid(random_x, random_t)
        #xs = np.concatenate([x_grid.reshape(-1,1), t_grid.reshape(-1, 1)], axis=1)
        xs = np.concatenate([random_x,random_t], axis=1)

        # collocation points    
        random_x = np.random.uniform(lb[0], ub[0], self.num_points).reshape(-1, 1)
        random_t = np.random.uniform(lb[1], ub[1], self.num_points).reshape(-1, 1)
        xf = np.concatenate([random_x, random_t], axis=1)

        # make them into tensors
        xf = torch.tensor(xf).float().to(device)
        xs = torch.tensor(xs).float().to(device)
        
        xs.requires_grad = True
        # predictions seed
        prediction_seed = model(xs)
        print('xs', xs.shape)
        
        loss_seed = pde(xs, prediction_seed)
        losses_xf = torch.zeros_like(xf)
        dist = torch.cdist(xf, xs, p=2)
        knn = dist.topk(1, largest=False)
        losses_xf = loss_seed[knn.indices[:, 0]]
        q_model = torch.softmax(losses_xf, dim=0)
        indicies_new = torch.multinomial(q_model[:, 0], self.num_points, replacement=True)

        self.xf = xf[indicies_new]
        self.weight = q_model[indicies_new]

        # plt.figure(figsize=(10,10))
        # plt.gca().set_title('Adaptive Sampling')
        # x_domain = np.linspace(lb[0], ub[0], 200)
        # t_domain = np.linspace(lb[1], ub[1], 200)
        # X, T = np.meshgrid(x_domain, t_domain)
        # X = X.reshape(-1,1)
        # T = T.reshape(-1,1)
        # #x_field = np.concatenate([X,T],axis=1)
        # x_field = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

        # field_prediction = model(torch.tensor(x_field).float().to(device))
        # field_prediction = field_prediction.detach().cpu().numpy()
        # field_prediction = field_prediction.reshape(200, 200)
        # plt.pcolormesh(t_domain, x_domain, field_prediction.T)
        # #x_train = x_train.cpu().numpy()
        # xs = xs.detach().cpu().numpy()
        # plt.scatter(xs[:, 1],xs[:, 0], c='b', marker='x', s=50, alpha=0.5)
        # #plt.scatter(x_train[:, 0], x_train[:, 1], c='r', marker='x', s=1, alpha=0.5)
        # xff=self.xf.cpu().numpy()
        # plt.scatter(xff[:, 1], xff[:, 0], c='r', marker='x', s=50, alpha=0.5)
        # plt.show()
        

        plt.figure(figsize=(10,10))
        t = np.linspace(0,ub[1],200).flatten()[:, None]
        x = np.linspace(0,ub[0],200).flatten()[:, None]
        X, T = np.meshgrid(x, t)

        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        pred = model(Tensor(X_star).to(device))
        pred_u = pred.detach().cpu().numpy()

        H_pred = pred_u.reshape(X.shape)
        plt.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                    extent= [lb[1], ub[1], lb[0], ub[0]],
                    origin='lower', aspect='auto')
        xs = xs.detach().cpu().numpy()
        plt.scatter(xs[:, 1],xs[:, 0], c='g', marker='x', s=50, alpha=0.5)
        xff=self.xf.cpu().numpy()
        plt.scatter(xff[:, 1], xff[:, 0], c='r', marker='x', s=50, alpha=0.5)
        plt.title('PINN')
        plt.ylabel('x (cm)')
        plt.xlabel('t (seconds)')
        plt.colorbar().set_label('Temperature (°C)')
        plt.show()
      return self.xf, self.weight


