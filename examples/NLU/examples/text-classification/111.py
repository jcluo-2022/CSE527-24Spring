def modify_matrix(lora_state_dict, model):
    num_layers = len(model.roberta.encoder.layer)
    for i in range(num_layers):
        key_A = f'model.roberta.encoder.layer[{i}].attention.self.query.lora_A'
        key_B = f'model.roberta.encoder.layer[{i}].attention.self.query.lora_B'

        if key_A in lora_state_dict and key_B in lora_state_dict:
            A = lora_state_dict[key_A]
            B = lora_state_dict[key_B]

            lora_update = torch.mm(A, B)

            query_weights = model.roberta.encoder.layer[i].attention.self.query.weight
            query_weights.data += lora_update[:query_weights.size(0), :query_weights.size(1)]

    return new_dict