## Notes:
1. Normalised state representation using the mean and variance of U(R,D-R) where D is the dimension of the square room. 
2. Fixed the heatmap visualisation tool accordingly so the critic is only evaluated for normalised values. 
3. Use elu. 
4. Fix the heatmap function. 
5. Use the following squared loss:

self.train_critic_and_source = dual_opt("critic", "source", self.squared_loss,\
                                                self.slow_optimizer)

## Training parameters:
horizon = 3
seed = 42
bound = 1.0
iters = 10000 
batch_size = 32
lr = 0.01
prob = 0.5