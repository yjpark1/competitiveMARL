# environment control
max_episode_len = 25
num_episodes = 40000
reward_factor = 1

# learning control
is_training = True
tau = 1e-2
actor_learning_rate = None
critic_learning_rate = None
batch_size = 1024
warmup_steps = batch_size
update_rate = 100
max_nb_steps = 1e+6

# verbose control
display = False
save_rate = 1000
exp_name = 'model_'

# train model path appendix
appx = 'env_partial/proposed+gumbel/'

# actions
# 0: nothing
# 1: left
# 2: right
# 3: down
# 4: up

