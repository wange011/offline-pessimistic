from configs import configdict

def get_config():

    config = configdict.ConfigDict()

    config.main = configdict.ConfigDict()
    config.main.alg = "safari"
    config.main.scenario_name = "simple_spread_n15"
    config.main.gym_env = False
    config.main.dir_main = "simple_spread_n15/safari"
    config.main.dir_data = "simple_spread_n15"
    config.main.seed = 0

    config.main.n_envs = 1
    config.main.n_eval_episodes = 20

    config.alg = configdict.ConfigDict()

    # Iterations
    config.alg.N = 500
    config.alg.H = 25

    # Parameters
    config.alg.alpha = 1
    config.alg.beta = 1
    config.alg.lamb = .98

    config.alg.sigma_kernel = 1

    return config
