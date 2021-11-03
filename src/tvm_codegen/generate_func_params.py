

def generate_block_fusion_params():
    params = "d_inputs[0],"
    num_weights = 64
    for i in range(num_weights):
        params += (" {}, {},".format("d_weights[{}]".format(i), \
            "d_bias[{}]".format(i)))
    for i in range(num_weights):
        params += ("{}, ".format("d_outputs[{}]".format(i)))
    return params


if __name__ == "__main__":
    print(generate_block_fusion_params())
