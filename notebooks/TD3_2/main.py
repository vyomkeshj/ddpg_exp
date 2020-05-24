import argparse
import os

import gym
import numpy as np
import torch

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
from notebooks.TD3_2 import TD3, replay_buffer
from notebooks.TD3_2.experience_replay_buffer import HindsightExperienceReplayBuffer
from notebooks.TD3_2.replay_buffer import ReplayBuffer


def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = env_name;
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over ", eval_episodes +" episodes: ", avg_reward)
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=4e5, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()



	file_name = "{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print("Policy: " ,args.policy, "Seed:", args.seed)
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	gym.envs.register(
		id='CustomEnv-v333',
		entry_point='envs.ur3_simenv:RobotEnv')

	env = gym.make("CustomEnv-v333")

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
		#policy.load("./models/model-her")
	replay_buffer = HindsightExperienceReplayBuffer(state_dim, action_dim)

	# Evaluate untrained policy
	#evaluations = [eval_policy(policy, env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0


	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)

		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		epoch_end = False if episode_timesteps < env._max_episode_steps else True
		# Store data in replay buffer,
		# todo do the HER modification here
		#print("adding state to replay buffer", state)
		if bool(done) | epoch_end:
			reward = 1
			replay_buffer.add(state, action, next_state, reward, 1)	#goes to temp buffer. todo: should done bool be 1
			replay_buffer.move_to_replay(next_state[2:5]) #moves to replay and flushes temp fixme: does next state have the final state?
		else:
			replay_buffer.add(state, action, next_state, reward, done_bool)	#goes to temp buffer

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if bool(done) | epoch_end:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print("Total T:", {t+1}," Episode Num:", {episode_num+1}," Episode T:", {episode_timesteps}," Reward:", {episode_reward})
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			policy.save("./models/model-her")
