from pathlib import Path
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import configure_logger


def dqn_solve(
    pomdp,
    total_timesteps=25_000_000,
    learning_rate=3e-5,
    seed=0,
    log_dir=None,
    log_name=None,
    **kwargs,
):
    breakpoint()
    logger = configure_logger(tensorboard_log=log_dir, tb_log_name=log_name or 'DQN')
    if log_dir:
        pomdp = Monitor(pomdp, logger.get_dir() + '/monitor.csv')
        Path(log_dir).mkdir(exist_ok=True)
    
    eval_callback = EvalCallback(
        pomdp, best_model_save_path=logger.get_dir(), log_path=logger.get_dir(),
        deterministic=True, eval_freq=16000, n_eval_episodes=50
    )
    policy = DQN(
        'MlpPolicy', pomdp, learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=[128, 128]), seed=seed
    )
    policy.set_logger(logger)
    policy.learn(total_timesteps=total_timesteps, callback=eval_callback)
    return policy


def ppo_solve(
    pomdp,
    total_timesteps=25_000_000,
    learning_rate=3e-4,
    use_lstm=False,
    seed=0,
    log_dir=None,
    log_name=None,
):
    if use_lstm:
        from sb3_contrib.ppo_recurrent import MlpLstmPolicy, RecurrentPPO
        policy = RecurrentPPO(
            MlpLstmPolicy, pomdp, policy_kwargs=dict(lstm_hidden_size=32),
            ent_coef=0.011, learning_rate=learning_rate, n_steps=256, seed=seed, tensorboard_log=log_dir
        )
    else:
        policy = PPO(
            'MlpPolicy', pomdp, learning_rate=learning_rate, n_steps=1024,
            policy_kwargs=dict(net_arch=[256, 256]), seed=seed, tensorboard_log=log_dir
        )
    
    logger = configure_logger(tensorboard_log=log_dir, tb_log_name=log_name or 'PPO')
    eval_callback = EvalCallback(
        pomdp, best_model_save_path=logger.get_dir(), log_path=logger.get_dir(),
        deterministic=True, eval_freq=16000, n_eval_episodes=50
    )
    policy.set_logger(logger)
    policy.learn(total_timesteps=total_timesteps, callback=eval_callback)
    return policy
