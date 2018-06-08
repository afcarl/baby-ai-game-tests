from babyai.levels import levels
from babyai.levels.instrs import Instr, Object

import gym

from collections import OrderedDict


from copy import deepcopy
import random


from gym_minigrid.envs import Key, Ball, Box
from babyai.levels.instr_gen import gen_instr_seq, gen_object, gen_surface
from babyai.levels.verifier import InstrSeqVerifier, InstrVerifier, OpenVerifier, PickupVerifier


class RoomGridLevelRR:

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we've completed the intermediary steps
        for i, verifierRR in enumerate(self.verifierRRs):
            if verifierRR.step() is True and self.completed[i] is False:
                reward = self.intermediate_rewards[i] * (1 + sum(self.completed))
                self.completed[i] = True
        print(self.completed)

        # If we've successfully completed the mission
        if self.verifier.step() is True:
            done = True
            reward = 5 * self._reward()

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.verifier = InstrSeqVerifier(self, self.instrs)

        # Recreate the verifierS of intermediary steps
        self.verifierRRs = [InstrSeqVerifier(self, [instrRR]) for instrRR in self.instrRRs]
        self.completed = [False] * len(self.verifierRRs)
        return obs


class Level_UnlockRR(levels.Level_Unlock, RoomGridLevelRR):
    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        key, _ = self.add_object(1, 1, 'key', door.color)
        if self.distractors:
            self.add_distractors(num_distractors=3, room_i=1, room_j=1)
        self.place_agent(1, 1)
        self.instrRRs = [Instr(action="pickup", object=Object(key.type))]
        self.instrs = [Instr(action="open", object=Object(door.type))]


class Level_FindObjS5RR(levels.Level_FindObjS5, RoomGridLevelRR):
    def gen_mission(self):
        # Add a random object to a random room
        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(i, j)
        self.place_agent(1, 1)
        self.list_of_doors = self.connect_all()
        # print(self.list_of_doors)
        self.instrRRs = [Instr(action="open", object=Object(door.type, door.color)) for door in self.list_of_doors]
        self.instrs = [Instr(action="pickup", object=Object(obj.type))]


class Level_HiddenKeyCorridorRR(RoomGridLevelRR, levels.Level_HiddenKeyCorridor):
    def gen_mission(self):
        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, 3)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        key, _ = self.add_object(0, self._rand_int(0, 3), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, 1)

        # Make sure all rooms are accessible
        self.list_of_doors = self.connect_all()

        self.instrRRs = [Instr(action="open", object=Object(door.type, door.color)) for door in self.list_of_doors]
        self.intermediate_rewards = [0.05] * len(self.instrRRs)
        self.instrRRs.append(Instr(action="pickup", object=Object(key.type)))
        self.intermediate_rewards.append(0.2)
        self.instrs = [Instr(action="pickup", object=Object(obj.type))]


level_dict = OrderedDict()


for global_name in sorted(list(globals().keys())):
    if not global_name.startswith('Level_'):
        continue

    module_name = __name__
    level_name = global_name.split('Level_')[-1]
    level_class = globals()[global_name]


    # Register the levels with OpenAI Gym
    gym_id = 'BabyAI-%s-v0' % (level_name)
    entry_point = '%s:%s' % (module_name, global_name)
    gym.envs.registration.register(
        id=gym_id,
        entry_point=entry_point,
    )

    # Add the level to the dictionary
    level_dict[level_name] = level_class

    # Store the name and gym id on the level class
    level_class.level_name = level_name
    level_class.gym_id = gym_id