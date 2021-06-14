import torch
import torch.nn as nn

class FingerNet(nn.Module):
    def __init__(self, lb, ub,numFeatures = 500, numLayers = 8, activation = torch.relu, normalize=True):
        torch.manual_seed(1234)
        super(FingerNet, self).__init__()

        self.numFeatures = numFeatures
        self.numLayers = numLayers 
        self.lin_layers = nn.ModuleList()
        self.lb = torch.tensor(lb).float()
        self.ub = torch.tensor(ub).float()
        self.activation = activation
        self.normalize = normalize
        self.init_layers()


    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """
        self.in_x = nn.ModuleList()
        self.in_y = nn.ModuleList()
        self.in_z = nn.ModuleList()
        self.in_t = nn.ModuleList()
        lenInput = 1
        noInLayers = 3

        self.in_x.append(nn.Linear(lenInput,self.numFeatures))
        for _ in range(noInLayers):
            self.in_x.append(nn.Linear(self.numFeatures, self.numFeatures)) 

        self.in_y.append(nn.Linear(lenInput,self.numFeatures))
        for _ in range(noInLayers):
            self.in_y.append(nn.Linear(self.numFeatures, self.numFeatures)) 

        self.in_z.append(nn.Linear(lenInput,self.numFeatures))
        for _ in range(noInLayers):
            self.in_z.append(nn.Linear(self.numFeatures, self.numFeatures)) 

        self.in_t.append(nn.Linear(1,self.numFeatures))
        for _ in range(noInLayers):
            self.in_t.append(nn.Linear(self.numFeatures, self.numFeatures)) 

        for m in [self.in_x,self.in_y,self.in_z,self.in_t]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0)
        
        self.lin_layers.append(nn.Linear(4 * self.numFeatures,self.numFeatures))
        for i in range(self.numLayers):
            inFeatures = self.numFeatures
            self.lin_layers.append(nn.Linear(inFeatures,self.numFeatures))
        inFeatures = self.numFeatures
        self.lin_layers.append(nn.Linear(inFeatures,1))
        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x_in):
        if self.normalize:
            x = 2.0 * (x_in - self.lb) / (self.ub - self.lb) - 1.0

        x_inx = x_in[:,0].view(-1,1)
        x_iny = x_in[:,1].view(-1,1)
        x_inz = x_in[:,2].view(-1,1)
        x_int = x_in[:,3].view(-1,1)

        for i in range(0,len(self.in_x)):
            x_inx = self.in_x[i](x_inx)
            x_inx = self.activation(x_inx)

        
        for i in range(0,len(self.in_y)):
            x_iny = self.in_y[i](x_iny)
            x_iny = self.activation(x_iny)
      
        for i in range(0,len(self.in_z)):
            x_inz = self.in_z[i](x_inz)
            x_inz = self.activation(x_inz)
            
        for i in range(0,len(self.in_t)):
            x_int = self.in_t[i](x_int)
            x_int = self.activation(x_int)
   
        
        x = torch.cat([x_inx,x_iny,x_inz,x_int],1)


        for i in range(0,len(self.lin_layers)-1):
            x = self.lin_layers[i](x)
            x = self.activation(x)
        x = self.lin_layers[-1](x)
   
        return x
        
    
    def cuda(self):
        super(FingerNet, self).cuda()
        self.lb = self.lb.cuda()
        self.ub = self.ub.cuda()

    def cpu(self):
        super(FingerNet, self).cuda()
        self.lb = self.lb.cpu()
        self.ub = self.ub.cpu()
        
    def to(self, device):
        super(FingerNet,self).to(device)
        self.lb.to(device)
        self.ub.to(device)