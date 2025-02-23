from transformers.models.vit.modeling_vit import ViTConfig, ViTSelfAttention
import torch.nn.functional as F

from tlora.tensor_factorization import FactorizedTensor

class ModifiedViTSdpaSelfAttention(ViTSelfAttention):
    def __init__(self, config: ViTConfig, factorization: str = "lora", rank: int = 8):
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        # Freeze original parameters
        for param in [self.query.weight, self.key.weight, self.value.weight]:
            param.requires_grad = False

        # Initialize factorization through factory
        self.factorization = FactorizedTensor.create(
            factorization_type=factorization,
            hidden_size=config.hidden_size,
            rank=rank
        )

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        if output_attentions or head_mask is not None:
            return super().forward(hidden_states, head_mask, output_attentions)

        # Get adaptation deltas
        delta_q, delta_k, delta_v = self.factorization()

        # Efficient weight adaptation without in-place modification
        query = F.linear(hidden_states, self.query.weight + delta_q, self.query.bias)
        key = F.linear(hidden_states, self.key.weight + delta_k, self.key.bias)
        value = F.linear(hidden_states, self.value.weight + delta_v, self.value.bias)

        # Original attention computation remains unchanged
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        context = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False
        )

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return (context,)