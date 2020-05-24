from notebooks.TD3_2.replay_buffer import ReplayBuffer
import torch
import numpy as np

class HindsightExperienceReplayBuffer(object):

    def __init__(self, state_dim, action_dim):
        self.temp_storage = ReplayBuffer(state_dim, action_dim, max_size=int(1e3))
        self.replay_storage = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_final = np.zeros(3, dtype=float)

    def add(self, state, action, next_state, reward, done):
        self.temp_storage.add(state, action, next_state, reward, done)
        if(self.current_final ==  np.zeros(3, dtype=float)).all():        #initially, when the current_final is empty
            self.current_final = state[2:5]                         #initialise with first response todo: remove hardcoded indices
            #print("current final assigned as ", self.current_final)

    def sample(self, batch_size):
        return self.replay_storage.sample(batch_size)


    def move_to_replay(self, final_pos): # todo: compute the change to the her here
        state, action, next_state, reward, done = self.temp_storage.fetchAll()
        for istate, iaction, inext_state, ireward, idone in zip(state, action, next_state, reward, done):
            #print("State before ", istate)  #fixme: find where the zeros are coming from

            #print("New target  ", final_pos)

            istate[2] = istate[2] + self.current_final[0] - final_pos[0]        # state is targetpos - robotpos, subtract by targetpos (final before change) and
            istate[3] = istate[3] + self.current_final[1] - final_pos[1]
            istate[4] = istate[4] + self.current_final[2] - final_pos[2]

            inext_state[2] = inext_state[2] + self.current_final[0] - final_pos[0]
            inext_state[3] = inext_state[3] + self.current_final[1] - final_pos[1]
            inext_state[4] = inext_state[4] + self.current_final[2] - final_pos[2]
            #print("State after ", inext_state)

            self.replay_storage.add(istate.cpu(), iaction.cpu(), inext_state.cpu(), ireward.cpu(), idone.cpu())        #modify before
            self.temp_storage.flush()
            self.current_final = np.zeros(3, dtype=float)  # reset current final to store the final position for the next epoch

    def fetchAll(self):
            return self.replay_storage.fetchAll()