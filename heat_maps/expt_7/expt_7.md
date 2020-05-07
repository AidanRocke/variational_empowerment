## Notes:
1. Normalised state representation using the mean and variance of U(R,D-R) where D is the dimension of the square room. 
2. Fixed the heatmap visualisation tool accordingly so the critic is only evaluated for normalised values. 
3. Use elu. 
4. Used the net action as input to the source and decoder rather than just the last action. The rationale behind this is that
the neural network wouldn't be able to keep track of the net effect of actions otherwise. 

## Training parameters:
horizon = 4
seed = 42
bound = 1.0
iters = 10000 
batch_size = 50
lr = 0.01
prob = 0.8