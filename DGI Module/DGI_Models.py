import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True 
cudnn.benchmark = True 
import torch.nn.functional as F

from torch_geometric.nn import ARMAConv,ChebConv,ClusterGCNConv,EGConv,FeaStConv,FiLMConv,\
GATv2Conv,GENConv,GeneralConv,GraphConv,HypergraphConv,LEConv,MFConv,ResGatedGraphConv,\
SAGEConv,SGConv,SuperGATConv,TAGConv,TransformerConv,GCNConv,GraphUNet



## 1. ARMAConv    ##03
class ARMAmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(ARMAmodel, self).__init__()

        # [in_dim, num_hidden, out_dim] = hidden_dims
        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        
        self.conv1 = ARMAConv(in_channels, num_hidden)

        self.conv2 = ARMAConv(num_hidden, num_hidden)

        self.conv3 = ARMAConv(num_hidden, num_hidden)

        self.conv4 = ARMAConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    # def forward(self, features, edge_index): #num_nodes, num_edges
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        # h1 = Activation_f(self.conv1(features, edge_index,edge_attr=edge_attr))
        # h2 = Activation_f(self.conv2(h1,edge_index,edge_attr=edge_attr))
        # h3 = Activation_f(self.conv3(h2,edge_index,edge_attr=edge_attr))
        h1 = Activation_f(self.conv1(features, edge_index,edge_weight=edge_attr))
        h2 = Activation_f(self.conv2(h1,edge_index,edge_weight=edge_attr))
        h3 = Activation_f(self.conv3(h2,edge_index,edge_weight=edge_attr))
        h4 = self.conv4(h3,edge_index,edge_weight=edge_attr)
        h4 = self.eLU(h4)

        return h4 
    
## 2. ChebConv    ##05
class Chebmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(Chebmodel, self).__init__()

        # [in_dim, num_hidden, out_dim] = hidden_dims
        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        
        self.conv1 = ChebConv(in_channels, num_hidden,K=1)

        self.conv2 = ChebConv(num_hidden, num_hidden,K=1)

        self.conv3 = ChebConv(num_hidden, num_hidden,K=1)

        self.conv4 = ChebConv(num_hidden, out_channels,K=1)

        self.eLU = nn.ELU(out_channels) 
    # def forward(self, features, edge_index): #num_nodes, num_edges
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        # h1 = Activation_f(self.conv1(features, edge_index,edge_attr=edge_attr))
        # h2 = Activation_f(self.conv2(h1,edge_index,edge_attr=edge_attr))
        # h3 = Activation_f(self.conv3(h2,edge_index,edge_attr=edge_attr))
        h1 = Activation_f(self.conv1(features, edge_index,edge_weight=edge_attr))
        h2 = Activation_f(self.conv2(h1,edge_index,edge_weight=edge_attr))
        h3 = Activation_f(self.conv3(h2,edge_index,edge_weight=edge_attr))
        h4 = self.conv4(h3,edge_index,edge_weight=edge_attr)
        h4 = self.eLU(h4)

        return h4 
    
## 3. ClusterGCNConv    ##06    
class ClusterGCNmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(ClusterGCNmodel, self).__init__()

        # [in_dim, num_hidden, out_dim] = hidden_dims
        [in_channels, num_hidden, out_channels] = hidden_dims #3000 256 256  
        
        self.conv1 = ClusterGCNConv(in_channels, num_hidden,  
                             add_self_loops=False, bias=False) 

        self.conv2 = ClusterGCNConv(num_hidden, num_hidden, 
                             add_self_loops=False, bias=False)

        self.conv3 = ClusterGCNConv(num_hidden, num_hidden, 
                             add_self_loops=False, bias=False)

        self.conv4 = ClusterGCNConv(num_hidden, out_channels, 
                             add_self_loops=False, bias=False)

        self.eLU = nn.ELU(out_channels) 

    def forward(self, data):
        Activation_f = F.elu     
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr

        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)

        return h4 

## 4. EGConv    ##09 
class EGmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(EGmodel, self).__init__()

        # [in_dim, num_hidden, out_dim] = hidden_dims
        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        
        self.conv1 = EGConv(in_channels, num_hidden)

        self.conv2 = EGConv(num_hidden, num_hidden)

        self.conv3 = EGConv(num_hidden, num_hidden)

        self.conv4 = EGConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        # h1 = Activation_f(self.conv1(features, edge_index,edge_attr=edge_attr))
        # h2 = Activation_f(self.conv2(h1,edge_index,edge_attr=edge_attr))
        # h3 = Activation_f(self.conv3(h2,edge_index,edge_attr=edge_attr))
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)

        return h4 

## 5. FeaStConv    ##11
class FeaStmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(FeaStmodel, self).__init__()

        # [in_dim, num_hidden, out_dim] = hidden_dims
        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        
        self.conv1 = FeaStConv(in_channels, num_hidden)

        self.conv2 = FeaStConv(num_hidden, num_hidden)

        self.conv3 = FeaStConv(num_hidden, num_hidden)

        self.conv4 = FeaStConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        # h1 = Activation_f(self.conv1(features, edge_index,edge_attr=edge_attr))
        # h2 = Activation_f(self.conv2(h1,edge_index,edge_attr=edge_attr))
        # h3 = Activation_f(self.conv3(h2,edge_index,edge_attr=edge_attr))
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)

        return h4 

## 6. FiLMConv    ##12
class FiLMmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(FiLMmodel, self).__init__()

        # [in_dim, num_hidden, out_dim] = hidden_dims
        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        
        self.conv1 = FiLMConv(in_channels, num_hidden)

        self.conv2 = FiLMConv(num_hidden, num_hidden)

        self.conv3 = FiLMConv(num_hidden, num_hidden)

        self.conv4 = FiLMConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 

    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        # h1 = Activation_f(self.conv1(features, edge_index,edge_attr=edge_attr))
        # h2 = Activation_f(self.conv2(h1,edge_index,edge_attr=edge_attr))
        # h3 = Activation_f(self.conv3(h2,edge_index,edge_attr=edge_attr))
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)

        return h4 
    
## 7. GATv2Conv    ##16    
class GATv2model(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GATv2model, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 256 256  
        
        self.conv1 = GATv2Conv(in_channels, num_hidden, concat=False, 
                             add_self_loops=False, bias=False) 

        self.conv2 = GATv2Conv(num_hidden, num_hidden, concat=False,
                             add_self_loops=False, bias=False)

        self.conv3 = GATv2Conv(num_hidden, num_hidden, concat=False,
                             add_self_loops=False, bias=False)

        self.conv4 = GATv2Conv(num_hidden, out_channels, concat=False,
                             add_self_loops=False, bias=False)

        self.eLU = nn.ELU(out_channels) 

    def forward(self, data):
        Activation_f = F.elu# elu:1.0169; LeakyReLU: 0.9822; ReLU:0.9978; SELU: 1.6388; CELU: 1.0169; Sigmoid:2.8492; LogSigmoid:1.2184; Tanh:1.1520;      
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index,edge_attr=edge_attr))
        h2 = Activation_f(self.conv2(h1,edge_index,edge_attr=edge_attr))
        h3 = Activation_f(self.conv3(h2,edge_index,edge_attr=edge_attr))
        h4 = self.conv4(h3,edge_index,edge_attr=edge_attr)
        h4 = self.eLU(h4)

        return h4  
    
## 8. GENConv    ##19    
class GENmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GENmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 256 256  
        self.conv1 = GENConv(in_channels, num_hidden) 

        self.conv2 = GENConv(num_hidden, num_hidden)

        self.conv3 = GENConv(num_hidden, num_hidden)

        self.conv4 = GENConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 

    def forward(self, data):
        Activation_f = F.elu     
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)

        return h4    

## 9. GeneralConv    ##20 
class Generalmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(Generalmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims 
        self.conv1 = GeneralConv(in_channels, num_hidden) 

        self.conv2 = GeneralConv(num_hidden, num_hidden)

        self.conv3 = GeneralConv(num_hidden, num_hidden)

        self.conv4 = GeneralConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 

    def forward(self, data):
        Activation_f = F.elu     
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 

## 10. GraphConv    ##23 
class Graphmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(Graphmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = GraphConv(in_channels, num_hidden) 

        self.conv2 = GraphConv(num_hidden, num_hidden)

        self.conv3 = GraphConv(num_hidden, num_hidden)

        self.conv4 = GraphConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu     
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 

## 11. HypergraphConv    ##29 
class Hypergraphmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(Hypergraphmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = HypergraphConv(in_channels, num_hidden) 

        self.conv2 = HypergraphConv(num_hidden, num_hidden)

        self.conv3 = HypergraphConv(num_hidden, num_hidden)

        self.conv4 = HypergraphConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu     
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 

## 12. LEConv   ##30
class LEmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(LEmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = LEConv(in_channels, num_hidden) 

        self.conv2 = LEConv(num_hidden, num_hidden)

        self.conv3 = LEConv(num_hidden, num_hidden)

        self.conv4 = LEConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu   
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 

## 13. MFConv   ##33
class MFmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(MFmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = MFConv(in_channels, num_hidden) 

        self.conv2 = MFConv(num_hidden, num_hidden)

        self.conv3 = MFConv(num_hidden, num_hidden)

        self.conv4 = MFConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu   
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 

## 14. ResGatedGraphConv   ##41
class ResGatedGraphmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(ResGatedGraphmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = ResGatedGraphConv(in_channels, num_hidden) 

        self.conv2 = ResGatedGraphConv(num_hidden, num_hidden)

        self.conv3 = ResGatedGraphConv(num_hidden, num_hidden)

        self.conv4 = ResGatedGraphConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 

## 15. SAGEConv  ##44
class SAGEmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(SAGEmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = SAGEConv(in_channels, num_hidden) 

        self.conv2 = SAGEConv(num_hidden, num_hidden)

        self.conv3 = SAGEConv(num_hidden, num_hidden)

        self.conv4 = SAGEConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 

## 16. SGConv  ##45
class SGmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(SGmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = SGConv(in_channels, num_hidden) 

        self.conv2 = SGConv(num_hidden, num_hidden)

        self.conv3 = SGConv(num_hidden, num_hidden)

        self.conv4 = SGConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 

    def forward(self, data):
        Activation_f = F.elu   
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 
    
## 17. SuperGATConv  ##49   
class SuperGATmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(SuperGATmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = SuperGATConv(in_channels, num_hidden) 

        self.conv2 = SuperGATConv(num_hidden, num_hidden)

        self.conv3 = SuperGATConv(num_hidden, num_hidden)

        self.conv4 = SuperGATConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4   

## 18. TAGConv  ##50  
class TAGmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(TAGmodel, self).__init__()
        
        [in_channels, num_hidden, out_channels] = hidden_dims #3000 128 128 
        self.conv1 = TAGConv(in_channels, num_hidden) 

        self.conv2 = TAGConv(num_hidden, num_hidden)

        self.conv3 = TAGConv(num_hidden, num_hidden)

        self.conv4 = TAGConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu#     
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)
        return h4 

## 19. TransformerConv ##51
class Transformermodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(Transformermodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 256 256  
        self.conv1 = TransformerConv(in_channels, num_hidden) 

        self.conv2 = TransformerConv(num_hidden, num_hidden)

        self.conv3 = TransformerConv(num_hidden, num_hidden)

        self.conv4 = TransformerConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 

    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)


        return h4 #h4是最后的特征向量

## 20. GCNConv ##18
class GCNmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GCNmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 256 256  
        self.conv1 = GCNConv(in_channels, num_hidden) 

        self.conv2 = GCNConv(num_hidden, num_hidden)

        self.conv3 = GCNConv(num_hidden, num_hidden)

        self.conv4 = GCNConv(num_hidden, out_channels)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)


        return h4 

## 21. GraphUNet  ##modules 6
class GraphUNetmodel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GraphUNetmodel, self).__init__()

        [in_channels, num_hidden, out_channels] = hidden_dims #3000 256 256  
        self.conv1 = GraphUNet(in_channels, num_hidden,out_channels,depth=1) 

        self.conv2 = GraphUNet(num_hidden, num_hidden,out_channels,depth=1)

        self.conv3 = GraphUNet(num_hidden, num_hidden,out_channels,depth=1)

        self.conv4 = GraphUNet(num_hidden, num_hidden,out_channels,depth=1)

        self.eLU = nn.ELU(out_channels) 
    def forward(self, data):
        Activation_f = F.elu    
        features, edge_index, edge_attr= data.x, data.edge_index,data.edge_attr
        h1 = Activation_f(self.conv1(features, edge_index))
        h2 = Activation_f(self.conv2(h1,edge_index))
        h3 = Activation_f(self.conv3(h2,edge_index))
        h4 = self.conv4(h3,edge_index)
        h4 = self.eLU(h4)


        return h4 






















