## Notes:
1. Normalised state representation using the mean and variance of U(0,D) where D is the dimension of the square room. 
2. Fixed the heatmap visualisation tool accordingly so the critic is only evaluated for normalised values. 
3. Use relu instead of elu. 
4. Use dropout set to 0.5

## Training parameters:
horizon = 3
seed = 42
bound = 1.0
iters = 10000 
batch_size = 32
lr = 0.01
prob = 0.5