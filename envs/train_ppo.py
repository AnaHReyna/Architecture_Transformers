from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from carla_route_env import CarlaRouteEnv, EnvConfig

def make_env():
    cfg = EnvConfig()

    cfg.host = "127.0.0.1"
    cfg.port = 2000
    cfg.traffic_manager_port = 8000

    cfg.route_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/routes_devtest.xml" 
    # cfg.scenario_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/all_towns_traffic_scenarios1_3_4.json"
    cfg.scenario_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/all_towns_traffic_scenarios.json"

    # Use None para amostrar rotas aleatórias do arquivo
    # Use "20" para fixar uma rota e depurar
    # cfg.route_id = "20"
    cfg.route_id = None
    # cfg.route_towns = ["Town01", "Town03", "Town05"]
    cfg.route_town = "Town05"

    cfg.fixed_delta_seconds = 0.05
    cfg.max_episode_steps = 500
    cfg.target_speed_kmh = 25.0
    cfg.ego_filter = "vehicle.lincoln.mkz_2017"

    cfg.render_rgb_camera = True

    # Visualização
    cfg.spectator_follow = True
    cfg.spectator_height_m = 40.0
    cfg.spectator_rotate_with_ego = False

    cfg.show_bev = False
    cfg.bev_width = 800
    cfg.bev_height = 800
    cfg.bev_fov = 90
    cfg.bev_height_m = 35.0

    env = CarlaRouteEnv(cfg)
    return Monitor(env)


if __name__ == "__main__":

    env = make_env()

    model = PPO("MlpPolicy",
                env,
                verbose=1,
                device="cpu",
                n_steps=256,
                batch_size=64,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.005,
                clip_range=0.2,
                tensorboard_log="./tb_logs/",
                )

    model.learn(total_timesteps=50_000)
    model.save("ppo_carla_route_v1")

    env.close()



# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import CheckpointCallback

# from carla_route_env import CarlaRouteEnv, EnvConfig


# def make_env():
#     cfg = EnvConfig()

#     cfg.host = "127.0.0.1"
#     cfg.port = 2000
#     cfg.traffic_manager_port = 8000

#     cfg.route_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/routes_devtest.xml"
#     cfg.scenario_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/all_towns_traffic_scenarios1_3_4.json"

#     # Para depuração visual, prefira uma rota fixa:
#     # cfg.route_id = "20"
#     cfg.route_id = None

#     cfg.fixed_delta_seconds = 0.05
#     cfg.max_episode_steps = 500
#     cfg.target_speed_kmh = 25.0
#     cfg.ego_filter = "vehicle.lincoln.mkz_2017"

#     cfg.render_rgb_camera = False

#     # Visualização
#     cfg.spectator_follow = True
#     cfg.spectator_height_m = 40.0
#     cfg.spectator_rotate_with_ego = True

#     cfg.show_bev = True
#     cfg.bev_width = 800
#     cfg.bev_height = 800
#     cfg.bev_fov = 90
#     cfg.bev_height_m = 35.0

#     env = CarlaRouteEnv(cfg)
#     return Monitor(env)


# if __name__ == "__main__":
#     env = make_env()

#     checkpoint_callback = CheckpointCallback(
#         save_freq=5000,
#         save_path="./checkpoints/",
#         name_prefix="ppo_carla_route",
#     )

#     try:
#         model = PPO(
#             "MlpPolicy",
#             env,
#             verbose=1,
#             device="cpu",
#             n_steps=256,
#             batch_size=64,
#             learning_rate=3e-4,
#             gamma=0.99,
#             gae_lambda=0.95,
#             ent_coef=0.005,
#             clip_range=0.2,
#             tensorboard_log="./tb_logs/",
#         )

#         model.learn(
#             total_timesteps=50_000,
#             callback=checkpoint_callback,
#             progress_bar=True,
#         )
#         model.save("ppo_carla_route_v1")

#     finally:
#         env.close()





# Para acompanhar o TensorBoard: tensorboard --logdir ./tb_logs

