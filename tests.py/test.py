import neural_decoding.decoders as decoders

a = decoders.linear_decoders.ridge_regression(2, 2, {'lbda':0,'intercept':True})

decoders.linear_decoders

print(a)