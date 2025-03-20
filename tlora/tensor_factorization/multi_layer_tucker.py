import torch.nn as nn
import torch.nn.functional as F
import torch

from tlora.tensor_factorization import FactorizedTensor

class MultiLayerTucker(FactorizedTensor, factorization_type="multi_layer_tucker"):
    def __init__(self, hidden_size: int, rank, num_layers: int = 12):
        # Process rank into a tuple of 3 integers
        if isinstance(rank, int):
            rank = (rank, rank, rank, rank)
        elif isinstance(rank, tuple) and len(rank) != 4:
            raise ValueError("Rank must be an int or a tuple of 4 ints")
        
        super().__init__(hidden_size, rank)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        r_layer, r_mode, r_row, r_col = rank
        
        self.core = nn.Parameter(torch.zeros(r_layer, r_mode, r_row, r_col))  # Shape: [r_layer, r_mode, r_row, r_col]
        
        self.layer_factors = nn.Parameter(torch.zeros(num_layers, r_layer))
        self.mode_factors = nn.Parameter(torch.zeros(3, r_mode))
        self.row_factors = nn.Parameter(torch.zeros(hidden_size, r_row))
        self.col_factors = nn.Parameter(torch.zeros(hidden_size, r_col))

        # Orthogonal initialization for factors
        nn.init.orthogonal_(self.layer_factors)
        nn.init.orthogonal_(self.mode_factors)
        nn.init.orthogonal_(self.row_factors)
        nn.init.orthogonal_(self.col_factors)

    def forward(self, id_layer: int):
        print(f'forward called with id: {id_layer}')
        # Shape of `result`: [num_layers, 3, hidden_size, hidden_size]
        result = torch.einsum(
            "lmrc, dl, qm, hr, kc -> dqhk", 
            self.core,             
            self.layer_factors,      
            self.mode_factors,          
            self.row_factors,           
            self.col_factors            
        )

        delta_q = result[id_layer][0]
        delta_k = result[id_layer][1]
        delta_v = result[id_layer][2]
        
        return delta_q, delta_k, delta_v