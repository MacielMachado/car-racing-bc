from car_racing_interface import CarRacingInterface
from cart_racing import CarRacing
from data_precessing import DataHandler
from model import Model, Model2
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt

class Tester():
    def __init__(self, model, env, render=True, device="mps"):
        self.path = os.getcwd() + '/experiments/'
        self.device = device
        self.render = render
        self.model = model.to(device)
        self.env = env
        self.observations = []
        self.actions = []

    def run(self, save, time_in_s=1e100, name=''):
        # np.random.seed(3) # 1
        episode = 0
        reward_list = []  
        while episode < 20:
            reward = 0
            obs_orig = self.env.reset()
            tempo_inicial = time.time()
            counter = 0
            while counter < 1000:
                self.model.eval()
                obs = DataHandler().to_greyscale(obs_orig)
                obs = DataHandler().normalizing(obs)
                obs = DataHandler().stack_with_previous(np.expand_dims(obs, axis=0))
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                action = self.model(obs_tensor).to("cpu")
                if save: self.save_game(obs_orig, action.detach().numpy())
                obs_orig, new_reward, done, _ = self.env.step(action.detach().numpy()[0] * [1, 1, 1])
                reward += new_reward
                print(f"{name} - episode: {episode} - count: {counter} - reward: {reward}")
                counter += 1
                print(f'counter: {counter}')
                if self.render: self.env.render()
                if done or counter > 5000:
                    # reward_list.append(reward)
                    # counter = 0
                    # reward = 0
                    # self.env.reset()
                    break

                # tempo_decorrido = time.time() - tempo_inicial
                # if tempo_decorrido >= time_in_s:
                #     break
            episode += 1
            reward_list.append(reward)
        self.scatter_plot_reward(reward_list, name)

        if save:
            hash = str(int(time.time()))
            np.save(self.path+'states_'+hash+'.npy',
                    self.observations)
            np.save(self.path+'actions_'+hash+'.npy',
                    self.actions)
        self.env.close()

    def save_game(self, obs, action):
        '''
        '''
        os.makedirs(self.path, exist_ok=True)
        self.actions.append(action)
        self.observations.append(obs)

    def scatter_plot_reward(self, reward_list, name):
        plt.subplot()
        plt.scatter(range(len(reward_list)), reward_list)
        plt.axhline(y=900, color='r', linestyle='--', linewidth=2)
        plt.title(f"Reward Scatter {name} - Mean: {sum(reward_list)/len(reward_list):.2f}")
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.grid()
        path = "experiments/"
        os.makedirs(path, exist_ok=True)
        plt.savefig(path+"bc_epoch_"+str(name)+".png")
        plt.close()

if __name__ == '__main__':
    model = Model()
    model_path = './model_pytorch/ppo/'
    episodes = [90, 70, 50, 30, 10, 1]
    for ep in episodes:
        # version = model_path + 'model_stacked_900_plus_ep_'+f'{ep}'+'.pkl'
        version = model_path + 'model_ppo_68_ep_130.pkl'
        model.load_state_dict(torch.load(version))
        env = CarRacing()
        Tester(model=model,env=env).run(save=False,
                                        time_in_s=1*60*60,
                                        name='model_stacked')