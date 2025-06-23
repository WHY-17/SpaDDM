import torch
from tqdm import tqdm
import torch.nn.functional as F
from Multi_Diffusion.model import Multi_Omics_DDM
from Multi_Diffusion.preprocess import adjacent_matrix_preprocessing




class Train_SpatialDDM:
    def __init__(
            self,
            data,
            graph,
            datatype='SPOTS',
            device=torch.device('cpu'),
            random_seed=2024,
            learning_rate=0.0001,
            weight_decay=0.0000001,
            epochs=600,
            dim_input=3000,
            dim_output=64,
            weight_factors=[1, 5, 1, 1],

            ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.

        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.graph = graph.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors


        # dgl_graph
        self.adata_omics_1 = self.data['adata_omics_1']
        self.adata_omics_2 = self.data['adata_omics_2']
        self.dgl_omics_1 = self.graph['dgl_omics_1'].to(self.device)
        self.dgl_omics_2 = self.graph['dgl_omics_2'].to(self.device)
        self.n_cell_omics_1 = self.adata_omics_1.n_obs
        self.n_cell_omics_2 = self.adata_omics_2.n_obs

        # dimension of input feature
        self.dim_input_1 = self.adata_omics_1.obsm['feat'].shape[1]
        self.dim_input_2 = self.adata_omics_2.obsm['feat'].shape[1]
        self.dim_output = dim_output

        if self.datatype == 'SPOTS':
            self.epochs = 600
            self.weight_factors = [0.5, 0.5, 0.1, 0.5]

        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 700
            self.weight_factors = [0.5, 0.5, 0.1, 0.5]

        elif self.datatype == '10x':
            self.epochs = 600
            self.weight_factors = [0.5, 0.5, 0.1, 0.5]

        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs = 1200
            self.weight_factors = [0.5, 0.5, 0.1, 0.5]

        elif self.datatype == 'Visium CytAssist':
            self.epochs = 800
            self.weight_factors = [0.5, 0.4, 0.1, 0.1]

    def train(self):
        self.model = Multi_Omics_DDM(in_dim_omics_1=self.dim_input_1, in_dim_omics_2=self.dim_input_2,
                                     num_hidden_omics_1=self.dim_output, num_hidden_omics_2=self.dim_output,
                                     ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.dgl_omics_1, self.dgl_omics_2)

            self.loss_recon_omics_1 = results['loss_rec_omics_1']
            self.loss_recon_omics_2 = results['loss_rec_omics_2']

            # reconstruction loss of multi modalities combined
            self.loss_corr_omics_1 = F.mse_loss(self.dgl_omics_1.ndata['feat'], results['comb_recon_omics_1'])
            self.loss_corr_omics_2 = F.mse_loss(self.dgl_omics_2.ndata['feat'], results['comb_recon_omics_2'])

            print("rec_loss_omics_1:", self.loss_recon_omics_1, "rec_loss_omics_2:", self.loss_recon_omics_2,
                  "corr_loss_omics_1:", self.loss_corr_omics_1, "corr_loss_omics_2:", self.loss_corr_omics_2)

            loss = (self.weight_factors[0] * self.loss_recon_omics_1 + self.weight_factors[1] * self.loss_recon_omics_2 +
                    self.weight_factors[2] * self.loss_corr_omics_1 + self.weight_factors[3] * self.loss_corr_omics_2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.dgl_omics_1, self.dgl_omics_2)

        emb_omics_1 = F.normalize(results['emb_omics_1'], p=2, eps=1e-12, dim=1)
        emb_omics_2 = F.normalize(results['emb_omics_2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_combined'], p=2, eps=1e-12, dim=1)
        rec_omics_1 = F.normalize(results['comb_recon_omics_1'], p=2, eps=1e-12, dim=1)
        rec_omics_2 = F.normalize(results['comb_recon_omics_2'], p=2, eps=1e-12, dim=1)

        output = {'emb_omics_1': emb_omics_1.detach().cpu().numpy(),
                  'emb_omics_2': emb_omics_2.detach().cpu().numpy(),
                  'SpatialDDM': emb_combined.detach().cpu().numpy(),
                  # 'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  # 'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'rec_omics_1': rec_omics_1.detach().cpu().numpy(),
                  'rec_omics_2': rec_omics_2.detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()}
        return output



    """
    def extract_embeddings(self, T):
        with torch.no_grad():
            self.model.eval()
        emb_omics_1 = self.model.DDM_omics_1.embed(self.dgl_omics_1, self.dgl_omics_1.ndata['feat'], T)
        emb_omics_2 = self.model.DDM_omics_2.embed(self.dgl_omics_2, self.dgl_omics_2.ndata['feat'], T)
        emb_combined, alpha_omics_1_2 = self.model.atten_cross(emb_omics_1, emb_omics_2)
        emb_combined = F.normalize(emb_combined, p=2, eps=1e-12, dim=1)
        return emb_combined, alpha_omics_1_2
    """




