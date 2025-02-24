from copy import deepcopy

def replace_attention_layers(model, modified_layer, factorization, rank):
    for i in range(len(model.vit.encoder.layer)):
        original_layer = model.vit.encoder.layer[i]

        # Create new attention layer with factorization
        new_attention = modified_layer(model.config, factorization=factorization, rank=rank)

        # Copy original weights (Q/K/V) to new layer
        new_attention.query.load_state_dict(original_layer.attention.attention.query.state_dict())
        new_attention.key.load_state_dict(original_layer.attention.attention.key.state_dict())
        new_attention.value.load_state_dict(original_layer.attention.attention.value.state_dict())

        # Preserve output layer weights
        new_output = deepcopy(original_layer.attention.output)

        # Rebuild the attention module
        new_layer = deepcopy(original_layer)
        new_layer.attention.attention = new_attention
        new_layer.attention.output = new_output

        model.vit.encoder.layer[i] = new_layer
    return model