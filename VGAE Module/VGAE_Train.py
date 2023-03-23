import numpy as np
import pandas as pd
from tqdm import tqdm 
import scipy.sparse as sp 

from torch_geometric.nn import ARMAConv,ChebConv,ClusterGCNConv,EGConv,FeaStConv,FiLMConv,\
GATv2Conv,GENConv,GeneralConv,GraphConv,HypergraphConv,LEConv,MFConv,ResGatedGraphConv,\
SAGEConv,SGConv,SuperGATConv,TAGConv,TransformerConv,GCNConv,GraphUNet
from torch_geometric.utils import train_test_split_edges
import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True 
cudnn.benchmark = True 
import torch.nn.functional as F 
from torch_geometric.nn import VGAE 

from Paper2_functions import Adata2Torch_data

Conv_list = [ARMAConv,ChebConv,ClusterGCNConv,EGConv,FeaStConv,FiLMConv,\
 GATv2Conv,GENConv,GeneralConv,GraphConv,HypergraphConv,LEConv,MFConv,ResGatedGraphConv,\
 SAGEConv,SGConv,SuperGATConv,TAGConv,TransformerConv,GCNConv]


    
class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalEncoder,self).__init__()
        self.conv1 = Conv_list[19](in_channels, 2 * out_channels)
        self.conv_mu = Conv_list[19](2 * out_channels, out_channels)
        self.conv_logstd = Conv_list[19](2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index) 


def VGAE_Train(adata, 
                hidden_dims=[128, 128],
                # hidden_dims=[512, 256,128,30], 
                num_epochs=1000, 
                lr=1e-6, 
                key_added='SCGDL',
                gradient_clipping=5., 
                weight_decay=0.0001, 
                random_seed=0, 
                save_loss=False):
    """\
    Training graph attention auto-encoder.
    Parameters
    ----------
    adata: AnnData object of scanpy package.
    hidden_dims: The dimension of the encoder.
    n_epochs:Number of total epochs for training.
    lr: Learning rate for AdamOptimizer.
    key_added: The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping: Gradient Clipping. 
    weight_decay: Weight decay for AdamOptimizer.
    save_loss: If True, the training loss is saved in adata.uns['SCGDL_loss'].
    save_reconst_exp: If True, the reconstructed expression profiles are saved in adata.layers['SCGDL_ReX'].
    device: See torch.device.

    Returns
    -------
    AnnData
    """
    # seed_everything() 
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)

    if "Spatial_highly_variable_genes" in adata.var.columns:
        adata_Vars =  adata[:, adata.var['Spatial_highly_variable_genes']]
        print('Input Size using Spatial_variable_genes: ', adata_Vars.shape)
    elif 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
        print('Input Size using Highly_variable_genes: ', adata_Vars.shape)
    else:
        adata_Vars = adata
        print('Input Size using All genes list: ', adata_Vars.shape) 

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Please Compute Spatial Network using Spatial_Dis_Cal function first!") 

    ## Process the data
    data = Adata2Torch_data(adata_Vars) #Create torch.pyG data
    data = train_test_split_edges(data)
    
    ## Setting input and output channels
    hidden_dims = [data.x.shape[1]] + hidden_dims #hidden_dims = [3000,128,128]。
    in_channels, out_channels = hidden_dims[0], hidden_dims[2]
    
    ## Define the VGAE model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGAE(VariationalEncoder(in_channels, out_channels )).to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    data = data.to(device)
    
    #Training process using VGAE models.
    import time
    start_time = time.time()
    loss_list = []
    for epoch in tqdm(range(1, num_epochs+1)):
        model.train()
        optimizer.zero_grad() 
        z = model.encode(x, train_pos_edge_index) 
        loss = model.recon_loss(z,train_pos_edge_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss() 
        loss_list.append(float(loss))
        loss.backward()
        optimizer.step()
        if ((epoch)%1000) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, np.mean(loss_list)))
    end_time = time.time()
    print('Elapsed training time:{:.4f} seconds'.format((end_time-start_time)))   
     
    model.eval()
    z = model.encode(x, train_pos_edge_index)
    auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)          
    SCGDL_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = SCGDL_rep
    adata.uns["AUC"] = auc
    adata.uns["AP"] = ap
                
    if save_loss:
        adata.uns['Model_loss'] = loss_list 

    return adata 