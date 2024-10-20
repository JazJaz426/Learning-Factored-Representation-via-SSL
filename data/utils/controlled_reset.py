from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava
from minigrid.minigrid_env import MiniGridEnv
import pdb
import copy
import numpy as np
import random as random

class CustomEnvReset:

    def __init__(self, env_name):

        custom_reset = {'DoorKey': self._custom_reset_doorkey, 'LavaCrossing': self._custom_reset_lavacrossing, 'FourRooms': self._custom_reset_fourrooms}
        
        for k in custom_reset.keys():
            if k in env_name:
                self.factored_reset = custom_reset[k]

    def _custom_reset_doorkey(self, env, width, height, controlled_factors):
        
        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

        # Used locations
        used_locations = set([])

        # Create an empty grid
        env.unwrapped.grid = Grid(width, height)

        # Generate the surrounding walls
        env.unwrapped.grid.wall_rect(0, 0, width, height)

        # factor 1: control goal position 
        goal_pos = (controlled_factors['goal_pos'][0], controlled_factors['goal_pos'][1]) if 'goal_pos' in controlled_factors else (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
        if goal_pos not in used_locations:
            env.unwrapped.put_obj(Goal(), goal_pos[0], goal_pos[1])
            used_locations.add(goal_pos)

        
        # factor 2: control door position
        if 'door_pos' in controlled_factors:
            splitIdx = controlled_factors['door_pos'][0]
            doorIdx = controlled_factors['door_pos'][1]
        else:
            splitIdx = None; doorIdx = None
            while (splitIdx, doorIdx) in used_locations or (splitIdx is None or doorIdx is None):
                splitIdx = env.unwrapped._rand_int(2, width - 2)
                doorIdx = env.unwrapped._rand_int(1, height - 2)
        
        # factor 3: control door locked / unlocked
        # factor 4: control door open/closed
        env.unwrapped.grid.vert_wall(splitIdx, 0)
        door_locked = controlled_factors['door_locked'] if 'door_locked' in controlled_factors else True
        door_open = controlled_factors['door_open'] if 'door_open' in controlled_factors else False
        env.unwrapped.put_obj(Door("yellow", is_locked=door_locked, is_open=door_open), splitIdx, doorIdx)
        used_locations.add((splitIdx, doorIdx))

        # factor 5: control key position 
        # factor 6: control holding key
        # pdb.set_trace()
        if not (('door_locked' in controlled_factors and controlled_factors['door_locked'] is False) or  ('holding_key' in controlled_factors and controlled_factors['holding_key'] is True)):
            
            if 'key_pos' in controlled_factors:
                key_top = controlled_factors['key_pos'] 
                key_size = (1,1) 
            else:
                key_top = (0,0)
                key_size = (splitIdx, height)

            env.unwrapped.place_obj(obj=Key("yellow"), top= key_top, size= key_size)
        else:
            #need to set the agent property as holding key
            env.unwrapped.carrying = Key("yellow")

        # factor 7: control agent position
        agent_top = tuple(controlled_factors['agent_pos']) if 'agent_pos' in controlled_factors else (0,0)
        agent_size = (1,1) if 'agent_pos' in controlled_factors else (splitIdx, height)
        env.unwrapped.place_agent(top=agent_top, size=agent_size)
        
        #factor 8: control agent direction 
        if 'agent_dir' in controlled_factors:
            env.unwrapped.agent_dir = controlled_factors['agent_dir']
        else:
            env.unwrapped.agent_dir = env.unwrapped._rand_int(0, 4)

        env.unwrapped.mission = "use the key to open the door and then get to the goal"
        #reset the original rng after resetting env
        env.unwrapped.np_random = curr_rng

        return env


        
    def _custom_reset_fourrooms(self, env, width, height, controlled_factors):
        
        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng
        

        used_locations = set([])

        # Create the grid
        env.unwrapped.grid = Grid(width, height)

        # Generate the surrounding walls
        env.unwrapped.grid.horz_wall(0, 0)
        env.unwrapped.grid.horz_wall(0, height - 1)
        env.unwrapped.grid.vert_wall(0, 0)
        env.unwrapped.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    env.unwrapped.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, env.unwrapped._rand_int(yT + 1, yB))
                    env.unwrapped.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    env.unwrapped.grid.horz_wall(xL, yB, room_w)
                    pos = (env.unwrapped._rand_int(xL + 1, xR), yB)
                    env.unwrapped.grid.set(*pos, None)

        # factor 1: control agent position
        agent_top = tuple(controlled_factors['agent_pos']) if 'agent_pos' in controlled_factors else None
        agent_size = (1,1) if 'agent_pos' in controlled_factors else None
        if 'agent_pos' in controlled_factors and agent_top not in used_locations:
            
            env.unwrapped.place_agent(top=agent_top, size=agent_size)
            used_locations.add(agent_top)

        else:
            agent_pos = env.unwrapped.place_agent()
            used_locations.add(agent_pos)

        # pdb.set_trace()
        #factor 2: control the agent direction
        if 'agent_dir' in controlled_factors:
            env.unwrapped.agent_dir = controlled_factors['agent_dir']
        else:
            env.unwrapped.agent_dir = env.unwrapped._rand_int(0, 4)

        #factor 3: control the goal position
        if 'goal_pos' in controlled_factors and (tuple(controlled_factors['goal_pos']) not in used_locations):
            goal = Goal()
            env.unwrapped.put_obj(goal, *tuple(controlled_factors['goal_pos']))
            goal.init_pos, goal.cur_pos = tuple(controlled_factors['goal_pos']), tuple(controlled_factors['goal_pos'])
        else:
            env.unwrapped.place_obj(Goal())
        
        env.unwrapped.mission = "reach the goal"

        #reset the original rng after resetting env
        env.unwrapped.np_random = curr_rng
        return env


    def _custom_reset_lavacrossing(self, env, width, height, controlled_factors):

        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

        used_locations = set([])

        assert width >= 5 and height >= 5

        env.unwrapped.grid = Grid(width, height)

        env.unwrapped.grid.wall_rect(0, 0, width, height)

        # factor 1: agent position
        if 'agent_pos' in controlled_factors and tuple(controlled_factors['agent_pos']) not in used_locations:
            agent_pos = tuple(controlled_factors['agent_pos'])
        else:
            agent_pos = env.unwrapped.place_agent()
            
        env.unwrapped.agent_pos =  agent_pos
        used_locations.add(agent_pos)

        # factor 2: agent orientation
        env.unwrapped.agent_dir = controlled_factors['agent_dir'] if 'agent_dir' in controlled_factors else env.unwrapped._rand_int(0, 4)


        # factor 3: goal position
        if 'goal_pos' in controlled_factors and tuple(controlled_factors['goal_pos']) not in used_locations:
            goal_pos = tuple(controlled_factors['goal_pos'])
        else:
            goal_pos = None
            
            while goal_pos is None or goal_pos in used_locations:
                goal_pos = (env.unwrapped._rand_int(0, width-2), env.unwrapped._rand_int(0, height-2))

        
        env.unwrapped.goal_pos = goal_pos
        used_locations.add(goal_pos)
        env.unwrapped.put_obj(Goal(), *env.unwrapped.goal_pos)

        # Generate and store random gap position
        env.unwrapped.gap_pos = np.array(
            (
                env.unwrapped._rand_int(2, width - 2),
                env.unwrapped._rand_int(1, height - 1),
            )
        )

        # Place the obstacle wall
        env.unwrapped.grid.vert_wall(env.unwrapped.gap_pos[0], 1, height - 2, Lava)

        # Put a hole in the wall
        env.unwrapped.grid.set(*env.unwrapped.gap_pos, None)

        env.unwrapped.mission = (
            "avoid the lava and get to the green goal square"
        )

        #reset the original rng after resetting env
        env.unwrapped.np_random = curr_rng
        
        return env
    