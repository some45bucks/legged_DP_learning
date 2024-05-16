from envs.joy_stick_env import JoyStickEnv

from brax import envs

envs.register_environment('joy_stick_env', JoyStickEnv)

print('All environments registered successfully!')