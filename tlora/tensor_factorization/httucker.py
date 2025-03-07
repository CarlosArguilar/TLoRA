import torch.nn as nn
import torch.nn.functional as F
import torch

from tlora.tensor_factorization import FactorizedTensor


# class Tucker3FactorizedTensor(FactorizedTensor, factorization_type="httucker"):
#     def __init__(self, hidden_size: int, rank):
#         # Process rank: if int, convert to a tuple; if tuple, verify length.
#         if isinstance(rank, int):
#             rank = (rank, rank, rank)
#         elif isinstance(rank, tuple):
#             if len(rank) != 3:
#                 raise ValueError("Rank tuple must contain 3 values: (r_mode, r_row, r_col)")
#         else:
#             raise ValueError("rank must be either an int or a tuple of 3 ints")
        
#         # Call parent constructor with the processed rank tuple
#         super().__init__(hidden_size, rank)
#         r_mode, r_row, r_col = rank
        
#         # Core tensor (compressed representation)
#         self.core = nn.Parameter(torch.zeros(r_mode, r_row, r_col))
        
#         # Factor matrices
#         self.mode_factors = nn.Parameter(torch.zeros(3, r_mode))  # Q/K/V mode
#         self.row_factors = nn.Parameter(torch.zeros(hidden_size, r_row))  # Matrix rows
#         self.col_factors = nn.Parameter(torch.zeros(hidden_size, r_col))  # Matrix columns

#         # Orthogonal initialization for the factor matrices
#         nn.init.orthogonal_(self.mode_factors)
#         nn.init.orthogonal_(self.row_factors)
#         nn.init.orthogonal_(self.col_factors)

#     def forward(self):
#         """Reconstruction: core[m,r,c] x mode[m] x row[r] x col[c]"""
#         deltas = torch.einsum(
#             "mrc, qm, hr, kc -> qhk",
#             self.core,
#             self.mode_factors,
#             self.row_factors,
#             self.col_factors
#         )
#         return deltas[0], deltas[1], deltas[2]  # Return Q, K, V matrices


class HTTuckerFactorizedTensor(FactorizedTensor, factorization_type="httucker"):
    def __init__(self, hidden_size: int, rank, num_heads: int = 12):
        # Process rank: if int, convert to a tuple; if tuple, verify length.
        if isinstance(rank, int):
            rank = (rank, rank, rank)
        elif isinstance(rank, tuple):
            if len(rank) != 3:
                raise ValueError("Rank tuple must contain 3 values: (r_mode, r_row, r_col)")
        else:
            raise ValueError("rank must be either an int or a tuple of 3 ints")
        
        # Call parent constructor with the processed rank tuple
        super().__init__(hidden_size, rank)
        r_mode, r_row, r_col = rank
        self.hidden_size = hidden_size
        
        # Core tensor (compressed representation)
        self.qcore = nn.Parameter(torch.zeros(r_mode, r_row, r_col))
        print(f"qcore size: {self.qcore.size()}")
        
        # Factor matrices
        self.qmode_factors = nn.Parameter(torch.zeros(num_heads, r_mode))  # Q/K/V mode
        print(f"qmode_factors size: {self.qmode_factors.size()}")
        
        self.qrow_factors = nn.Parameter(torch.zeros(self.hidden_size, r_row))  # Matrix rows
        print(f"qrow_factors size: {self.qrow_factors.size()}")
        
        self.qcol_factors = nn.Parameter(torch.zeros(self.hidden_size/num_heads, r_col))  # Matrix columns
        print(f"qcol_factors size: {self.qcol_factors.size()}")

        # Core tensor (compressed representation)
        self.kcore = nn.Parameter(torch.zeros(r_mode, r_row, r_col))
        
        # Factor matrices
        self.kmode_factors = nn.Parameter(torch.zeros(num_heads, r_mode))  # Q/K/V mode
        self.krow_factors = nn.Parameter(torch.zeros(self.hidden_size, r_row))  # Matrix rows
        self.kcol_factors = nn.Parameter(torch.zeros(self.hidden_size/num_heads, r_col))  # Matrix columns

        # Core tensor (compressed representation)
        self.vcore = nn.Parameter(torch.zeros(r_mode, r_row, r_col))
        
        # Factor matrices
        self.vmode_factors = nn.Parameter(torch.zeros(num_heads, r_mode))  # Q/K/V mode
        self.vrow_factors = nn.Parameter(torch.zeros(self.hidden_size, r_row))  # Matrix rows
        self.vcol_factors = nn.Parameter(torch.zeros(self.hidden_size/num_heads, r_col))  # Matrix columns


        # Orthogonal initialization for the factor matrices
        nn.init.orthogonal_(self.qmode_factors)
        nn.init.orthogonal_(self.qrow_factors)
        nn.init.orthogonal_(self.qcol_factors)

        # Orthogonal initialization for the factor matrices
        nn.init.orthogonal_(self.kmode_factors)
        nn.init.orthogonal_(self.krow_factors)
        nn.init.orthogonal_(self.kcol_factors)

        # Orthogonal initialization for the factor matrices
        nn.init.orthogonal_(self.vmode_factors)
        nn.init.orthogonal_(self.vrow_factors)
        nn.init.orthogonal_(self.vcol_factors)

    def forward(self):
        """Reconstruction: core[m,r,c] x mode[m] x row[r] x col[c]"""
        deltas_q = torch.einsum(
            "mrc, qm, hr, kc -> hkq",
            self.qcore,
            self.qmode_factors,
            self.qrow_factors,
            self.qcol_factors
        )
        
        print(f"deltas_q size before reshape: {deltas_q.size()}")
        deltas_q = deltas_q.reshape(self.hidden_size, self.hidden_size)
        print(f"deltas_q size after reshape: {deltas_q.size()}")

        deltas_k = torch.einsum(
            "mrc, qm, hr, kc -> hkq",
            self.kcore,
            self.kmode_factors,
            self.krow_factors,
            self.kcol_factors
        ).reshape(self.hidden_size, self.hidden_size)

        deltas_v = torch.einsum(
            "mrc, qm, hr, kc -> hkq",
            self.vcore,
            self.vmode_factors,
            self.vrow_factors,
            self.vcol_factors
        ).reshape(self.hidden_size, self.hidden_size)


        return deltas_q, deltas_k, deltas_v  # Return Q, K, V matrices
