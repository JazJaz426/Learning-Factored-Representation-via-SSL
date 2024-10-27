from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
import pdb
import copy
import numpy as np
import random as random

class CustomEnvReset:

    def __init__(self, env_name, all_factors):

        custom_reset = {'DoorKey': self._custom_reset_doorkey, 'LavaCrossing': self._custom_reset_lavacrossing, 'FourRooms': self._custom_reset_fourrooms}
        
        for k in custom_reset.keys():
            if k in env_name:
                self.factored_reset = custom_reset[k]
        
        self.all_factors = set(all_factors)
    
    def check_valid_factors(self, controlled_factors):
            
            #populate empty set with all controlled factors to check
            occupied_locations = set([])

            for k in controlled_factors.keys():
                controlled_val = controlled_factors[k]

                if len(controlled_val) == 2:

                    #check 0: if the location is within bounds
                    if controlled_val[0] >= env.unwrapped.width and controlled_val[0] < 0:
                        return False
                    if controlled_val[1] >= env.unwrapped.height and controlled_val[1] < 0:
                        return False

                    #check 1: if the location is in the wall border
                    if isinstance(env.unwrapped.grid.get(controlled_val[0], controlled_val[1]), Wall):
                        return False
                    
                    #check 2: check if the location is within occupied locations
                    if tuple((controlled_val[0], cnotrolled_val[1])) in occupied_locations:
                        return False
                    
                    #other checks: for door add entire column to set & for any other position add only position the set
                    if k == 'door_pos':
                        for i in range(env.unwrapped.height):
                            occupied_locations.add((controlled_val[0],i))
                    
                    occupied_locations.add((controlled_val[0], controlled_val[1]))

            #check 3: specific to doorkey, if door is open and key is not held
            if ('door_open' in controlled_factors and controlled_factors['door_open']) and ('door_locked' in controlled_factors and controlled_factors['door_locked']):
                return False
            
            #check 4: specific to doorkey, if holding key and key_pos is part of controlled factors
            if ('holding_key' in controlled_factors and controlled_factors['holding_key']) and ('key_pos' in controlled_factors):
                return False
             
            return True

    def _custom_reset_doorkey(self, env, width, height, controlled_factors):

        #check if the controlled factors is valid or not
        valid = self.check_valid_factors(controlled_factors)

        if not valid:
            raise Exception(f'ERROR: the factors {controlled_factors} are not valid')
        
        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

        # Used locations
        used_locations = set([(0,0)])


        #all set factor values
        all_factors = {}

        #insert controlled factors in to the set of all factors
        for f in controlled_factors.keys():
            
            factor = controlled_factors[f]

            if len(factor) == 2:

                #add location to used set
                used_locations.add(tuple(factor))

                #add column for door
                if f == 'door_pos':
                    for i in range(env.unwrapped.height):
                        used_locations.add((controlled_val[0],i))
            

            all_factors[f] = factor
        

        #randomly set the factor values for all other factors: sample until they are not in used locations
        remaining_factors = self.all_factors - set(list(controlled_factors.keys()))

        for f in remaining_factors:

            if f == 'goal_pos':
                rand_goal_loc = (0,0)

                while (rand_goal_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_goal_loc[0], rand_goal_loc[1]), Wall)):
                    rand_goal_loc =  (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_goal_loc
                used_locations.add(rand_goal_loc)
            
            if f == 'door_pos':

                rand_door_loc = (0,0)

                while (rand_door_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_door_loc[0], rand_door_loc[1]), Wall)):
                    rand_door_loc = (env.unwrapped._rand_int(2, width - 2), env.unwrapped._rand_int(1, height - 2))
                
                all_factors[f] = rand_door_loc
                used_locations.add(rand_door_loc)
            
            if f == 'door_locked':

                all_factors[f] = True if random.random() <=0.5 else False
            
            if f == 'door_open':
                all_factors[f] = True if random.random() <= 0.5 else False
            
            if f == 'agent_pos':
                rand_agent_loc = (0,0)

                while (rand_agent_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_agent_loc[0], rand_agent_loc[1]), Wall)):
                    rand_agent_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_agent_loc
                used_locations.add(rand_agent_loc)

            if f == 'agent_dir':
                all_factors[f] = env.unwrapped._rand_int(0, 4)

            if f == 'holding_key':
                all_factors[f] = True if random.random()<=0.5 else False
            
            if f == 'key_pos':
                
                #set key location anywhere if not holding and door open
                if not all_factors['holding_key'] and all_factors['door_open']:
                    rand_key_loc = (0,0)

                    while (rand_key_loc  in used_locations) and isinstance(env.unwrapped.grid.get(rand_key_loc[0], rand_key_loc[1]), Wall):
                        rand_agent_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                #set key location to left half if not holding and door not open
                elif not all_factors['holding_key'] and not all_factors['door_open']:
                    rand_key_loc = (0,0)

                    while (rand_key_loc  in used_locations) and isinstance(env.unwrapped.grid.get(rand_key_loc[0], rand_key_loc[1]), Wall):
                        rand_agent_loc = (env.unwrapped._rand_int(1, all_factors['door_pos']-1), env.unwrapped._rand_int(1, height - 1))

                #set key location to none if holding
                elif all_factors['holding_key']:

                    rand_key_loc = (None, None)
                
                all_factors[f] = rand_key_loc
        
        # Create an empty grid
        env.unwrapped.grid = Grid(width, height)

        # Generate the surrounding walls
        env.unwrapped.grid.wall_rect(0, 0, width, height)

        # factor 1: add goal position 
        env.unwrapped.put_obj(Goal(), all_factors['goal_pos'][0], all_factors['goal_pos'][1])
        
        # factor 2, 3, 4: add door position, with locked/unlocked and open/closed settings
        splitIdx = all_factors['door_pos'][0]; doorIdx = all_factors['door_pos'][1]
        door_locked = all_factors['door_locked']
        door_open = all_factors['door_open']

        env.unwrapped.grid.vert_wall(splitIdx, 0)
        env.unwrapped.put_obj(Door("yellow", is_locked=door_locked, is_open=door_open), splitIdx, doorIdx)

        # factor 5: add key position and holding
        # factor 6: control holding key
        if all_factors['key_pos'] != (None, None):
            key_top = all_factors['key_pos']
            key_size = (1,1)
            env.unwrapped.place_obj(obj=Key("yellow"), top= key_top, size= key_size)
        
        else:
            env.unwrapped.carrying = Key("yellow")

        # factor 7, 8: add agent position and direction
        agent_top = all_factors['agent_pos']
        agent_size = (1,1)
        env.unwrapped.place_agent(top=agent_top, size=agent_size)
        env.unwrapped.agent_dir = all_factors['agent_dir']

        env.unwrapped.mission = "use the key to open the door and then get to the goal"
        
        #reset the original rng after resetting env
        env.unwrapped.np_random = curr_rng

        return env


        
    def _custom_reset_fourrooms(self, env, width, height, controlled_factors):

        
        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

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

        # Used locations
        used_locations = set([(0,0)])


        #all set factor values
        all_factors = {}



        #insert controlled factors in to the set of all factors
        for f in controlled_factors.keys():
            
            factor = controlled_factors[f]

            if len(factor) == 2:

                #add location to used set
                used_locations.add(tuple(factor))
            

            all_factors[f] = factor
        

        #randomly set the factor values for all other factors: sample until they are not in used locations
        remaining_factors = self.all_factors - set(list(controlled_factors.keys()))

        for f in remaining_factors:

            if f == 'goal_pos':
                rand_goal_loc = (0,0)

                while (rand_goal_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_goal_loc[0], rand_goal_loc[1]), Wall)):
                    rand_goal_loc =  (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_goal_loc
                used_locations.add(rand_goal_loc)
            
            
            if f == 'agent_pos':
                rand_agent_loc = (0,0)

                while (rand_agent_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_agent_loc[0], rand_agent_loc[1]), Wall)):
                    rand_agent_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_agent_loc
                used_locations.add(rand_agent_loc)

            if f == 'agent_dir':
                all_factors[f] = env.unwrapped._rand_int(0, 4)


        

        # factor 1 / 2: set agent position and direction
        agent_top = all_factors['agent_pos']
        agent_size = (1,1)
        env.unwrapped.place_agent(top=agent_top, size=agent_size)
        env.unwrapped.agent_dir = all_factors['agent_dir']
        

        #factor 3: set the goal position
        env.unwrapped.put_obj(Goal(), *all_factors['goal_pos'])
       
        env.unwrapped.mission = "reach the goal"

        #reset the original rng after resetting env
        env.unwrapped.np_random = curr_rng
        return env


    def _custom_reset_lavacrossing(self, env, width, height, controlled_factors):

        #change the random seed locally 
        curr_rng = env.unwrapped.np_random
        local_rng = np.random.default_rng(int(100*random.random()))
        env.unwrapped.np_random = local_rng

        

        assert width >= 5 and height >= 5

        # Used locations
        used_locations = set([(0,0)])


        #all set factor values
        all_factors = {}

        #insert controlled factors in to the set of all factors
        for f in controlled_factors.keys():
            
            factor = controlled_factors[f]

            if len(factor) == 2:

                #add location to used set
                used_locations.add(tuple(factor))
            

            all_factors[f] = factor
        

        #randomly set the factor values for all other factors: sample until they are not in used locations
        remaining_factors = self.all_factors - set(list(controlled_factors.keys()))

        for f in remaining_factors:

            if f == 'goal_pos':
                rand_goal_loc = (0,0)

                while (rand_goal_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_goal_loc[0], rand_goal_loc[1]), Wall)):
                    rand_goal_loc =  (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_goal_loc
                used_locations.add(rand_goal_loc)
            
            if f == 'agent_pos':
                rand_agent_loc = (0,0)

                while (rand_agent_loc in used_locations) and (isinstance(env.unwrapped.grid.get(rand_agent_loc[0], rand_agent_loc[1]), Wall)):
                    rand_agent_loc = (env.unwrapped._rand_int(1, width - 1), env.unwrapped._rand_int(1, height - 1))
                
                all_factors[f] = rand_agent_loc
                used_locations.add(rand_agent_loc)

            if f == 'agent_dir':
                all_factors[f] = env.unwrapped._rand_int(0, 4)

           

        env.unwrapped.grid = Grid(width, height)
        env.unwrapped.grid.wall_rect(0, 0, width, height)

        # factor 1/2: set agent position and orientation
        env.unwrapped.agent_pos =  all_factors['agent_pos']
        env.unwrapped.agent_dir = all_factors['agent_dir'] 


        # factor 3: set goal position
        env.unwrapped.goal_pos = all_factors['goal_pos']
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
    