import sys
import random
import copy
import torch
from collections import namedtuple, deque
import itertools
import math
import numpy as np 
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

class pon:
    """ struktora dla pionka"""
    def __init__(self, animal: int, player: int):
        self.animal = animal 
        self.player = player
    
    def replace(self, animal: int, player:int):
        self.animal = animal
        self.player = player
        
    def empty(self) -> bool:
        return self.animal == 0 and self.player == 2
        
class move_struct:
    """ struktura dla ruchu """
    def __init__(self, from_poss, shift):
        self.from_poss = from_poss
        self.shift = shift 
    
    def replace(self, from_poss, shift):
        self.from_poss = from_poss
        self.shift = shift 
        
class Board: 
    # zmienne pseudo globalne (PYTHON)
    NO_ANIMAL, RAT, CAT, DOG, WOLF, JAGUAR, TIGER, LION, ELEPHANT = range(9)
    FIELD, TRAP, RIVER, DEN_P1, DEN_P2 = range(5)
    PRAND, PSMART, NO_PLAYER = range(3)

    # przydatne constanty
    n, m = 9, 7
    TILE_TYPE = [
        [FIELD, FIELD, TRAP, DEN_P1, TRAP, FIELD, FIELD],
        [FIELD, FIELD, FIELD, TRAP, FIELD, FIELD, FIELD],
        [FIELD, FIELD, FIELD, FIELD, FIELD, FIELD, FIELD],
        [FIELD, RIVER, RIVER, FIELD, RIVER, RIVER, FIELD],
        [FIELD, RIVER, RIVER, FIELD, RIVER, RIVER, FIELD],
        [FIELD, RIVER, RIVER, FIELD, RIVER, RIVER, FIELD],
        [FIELD, FIELD, FIELD, FIELD, FIELD, FIELD, FIELD],
        [FIELD, FIELD, FIELD, TRAP, FIELD, FIELD, FIELD],
        [FIELD, FIELD, TRAP, DEN_P2, TRAP, FIELD, FIELD]
    ]
    
    rand_trybe = 0
    last_reward = 0
    inapropriate_move = False
    shifts = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    max_player, min_player = 1, 0
    global model, device
    
    def __init__(self, player: int, board: list[list[pon]], animal_poss: list[list[tuple]], rand_trybe: int):
        """ konstruktor """
        self.player = player 
        self.board = board 
        self.animal_poss = animal_poss
        self.rand_trybe = rand_trybe
        self.last_reward = 0
        
        self.max_player = player
        self.min_player = player^1
        
    
    def print_board(self):
        """ wypisuje plansze """
        for i in range(self.n):
            for j in range(self.m):
                match self.board[i][j].animal:
                    case self.NO_ANIMAL: print('.', end='', sep='')
                    case self.RAT: print('R' if self.board[i][j].player == self.PSMART else 'r', end='', sep='')
                    case self.CAT: print('C' if self.board[i][j].player == self.PSMART else 'c', end='', sep='')
                    case self.DOG: print('D' if self.board[i][j].player == self.PSMART else 'd', end='', sep='')
                    case self.WOLF: print('W' if self.board[i][j].player == self.PSMART else 'w', end='', sep='')
                    case self.JAGUAR: print('J' if self.board[i][j].player == self.PSMART else 'j', end='', sep='')
                    case self.TIGER: print('T' if self.board[i][j].player == self.PSMART else 't', end='', sep='')
                    case self.LION: print('L' if self.board[i][j].player == self.PSMART else 'l', end='', sep='')
                    case self.ELEPHANT: print('E' if self.board[i][j].player == self.PSMART else 'e', end='', sep='')
            print()
        print()
    
    def at(self, x, y):
        # dodawanie krotek
        return (x[0] + y[0], x[1] + y[1])

    def valid_coord(self, poss):
        # sprawdzenie czy pozycja jest git
        return 0 <= poss[0] < self.n and 0 <= poss[1] < self.m
    
    def generate_forward_moves(self): #tylko dla random player
        possible_moves = []
        my_den = (0, 3)
        for animal in range(1, 9):
            if self.animal_poss[self.player][animal] == None: continue
            
            shift = [1, 0] #tylko do przodu 
            poss = self.at(self.animal_poss[self.player][animal], shift)
            
            if not self.valid_coord(poss): continue
            if self.board[poss[0]][poss[1]].animal != self.NO_ANIMAL and self.board[poss[0]][poss[1]].player == self.player: continue
            if poss == my_den: continue
            if animal != self.RAT and animal != self.TIGER and animal != self.LION and self.TILE_TYPE[poss[0]][poss[1]] == self.RIVER: continue
            
            match animal:
                case self.RAT:
                    if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and (self.board[poss[0]][poss[1]].animal > self.RAT and self.board[poss[0]][poss[1]].animal != self.ELEPHANT): continue
                    if self.TILE_TYPE[self.animal_poss[self.player][animal][0]][self.animal_poss[self.player][animal][1]] == self.RIVER and self.TILE_TYPE[poss[0]][poss[1]] != self.RIVER and self.board[poss[0]][poss[1]].animal: continue
                    possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                    
                case self.CAT:
                    if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.CAT: continue
                    possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                    
                case self.DOG:
                    if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.DOG: continue
                    possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                    
                case self.WOLF:
                    if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.WOLF: continue
                    possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                    
                case self.JAGUAR:
                    if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.JAGUAR: continue
                    possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                    
                case self.TIGER:
                    if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.TIGER: continue
                    if self.TILE_TYPE[poss[0]][poss[1]] == self.RIVER:
                        shift = self.make_jump_over_water(poss, animal, shift)
                        if shift == None: continue
                    possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                    
                case self.LION:
                    if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.LION: continue
                    if self.TILE_TYPE[poss[0]][poss[1]] == self.RIVER:
                        shift = self.make_jump_over_water(poss, animal, shift)
                        if shift == None: continue
                    possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                    
                case self.ELEPHANT:
                    if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal == self.RAT: continue
                    possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
        
        return possible_moves
        
    
    def generate_possible_moves(self):
        possible_moves = []
        my_den = (0, 3) if self.player == self.PRAND else (8, 3)
        for animal in range(1, 9):
            if self.animal_poss[self.player][animal] == None: continue
            for shift in self.shifts:
                poss = self.at(self.animal_poss[self.player][animal], shift)
                
                if not self.valid_coord(poss): continue
                if self.board[poss[0]][poss[1]].animal != self.NO_ANIMAL and self.board[poss[0]][poss[1]].player == self.player: continue
                if poss == my_den: continue
                if animal != self.RAT and animal != self.TIGER and animal != self.LION and self.TILE_TYPE[poss[0]][poss[1]] == self.RIVER: continue
                
                match animal:
                    case self.RAT:
                        if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and (self.board[poss[0]][poss[1]].animal > self.RAT and self.board[poss[0]][poss[1]].animal != self.ELEPHANT): continue
                        if self.TILE_TYPE[self.animal_poss[self.player][animal][0]][self.animal_poss[self.player][animal][1]] == self.RIVER and self.TILE_TYPE[poss[0]][poss[1]] != self.RIVER and self.board[poss[0]][poss[1]].animal: continue
                        possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                        
                    case self.CAT:
                        if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.CAT: continue
                        possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                        
                    case self.DOG:
                        if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.DOG: continue
                        possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                        
                    case self.WOLF:
                        if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.WOLF: continue
                        possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                        
                    case self.JAGUAR:
                        if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.JAGUAR: continue
                        possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                        
                    case self.TIGER:
                        if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.TIGER: continue
                        if self.TILE_TYPE[poss[0]][poss[1]] == self.RIVER:
                            shift = self.make_jump_over_water(poss, animal, shift)
                            if shift == None: continue
                        possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                        
                    case self.LION:
                        if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal > self.LION: continue
                        if self.TILE_TYPE[poss[0]][poss[1]] == self.RIVER:
                            shift = self.make_jump_over_water(poss, animal, shift)
                            if shift == None: continue
                        possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
                        
                    case self.ELEPHANT:
                        if self.TILE_TYPE[poss[0]][poss[1]] != self.TRAP and self.board[poss[0]][poss[1]].animal == self.RAT: continue
                        possible_moves.append(move_struct(self.animal_poss[self.player][animal], shift))
        
        return possible_moves
                        
    def make_jump_over_water(self, poss, animal, shift):
        new_shift = copy.copy(shift) 
        while self.TILE_TYPE[poss[0]][poss[1]] == self.RIVER:
            if self.board[poss[0]][poss[1]].animal > 0:
                new_shift = None # trafiliśmy na szczura
                return new_shift
            
            poss = (poss[0] + shift[0], poss[1] + shift[1])
            new_shift[0] += shift[0]
            new_shift[1] += shift[1]
        
        if self.board[poss[0]][poss[1]].player == self.player or self.board[poss[0]][poss[1]].animal > animal: new_shift = None
        return new_shift
        
    def do_move(self, move: move_struct): #zwraca reward i terminated
        global sum_lenght, sum_deads, sum_kils
        sum_lenght += 1
        if self.PSMART == self.player: self.last_reward -= 1
        if move == None:
            self.last_reward -= 1000
            return
        
        animal = self.board[move.from_poss[0]][move.from_poss[1]].animal
        new_poss = self.at(move.from_poss, move.shift)
        
        if self.board[new_poss[0]][new_poss[1]].animal:
            self.animal_poss[self.player^1][self.board[new_poss[0]][new_poss[1]].animal] = None
            if self.player == self.PSMART: self.last_reward += 25; sum_kils += 1 #zbicie pionka przeciwnika
            else: self.last_reward -= 1000; sum_deads += 1
        
        self.board[move.from_poss[0]][move.from_poss[1]].replace(self.NO_ANIMAL, self.NO_PLAYER)
        self.board[new_poss[0]][new_poss[1]].replace(animal, self.player)
        self.animal_poss[self.player][animal] = new_poss
        if self.player == self.PSMART and new_poss == (0, 3):
            self.last_reward += 1000
        
    def game_over(self):
        if self.board[0][3].animal or self.board[8][3].animal or self.inapropriate_move: return 1
        return len(self.generate_possible_moves()) == 0
    
    def make_random_move(self):
        possible_moves = []
        if self.rand_trybe == 0: #gramy randomowo
            possible_moves = self.generate_possible_moves()
        else: #preferujemy ruchy do przodu
            possible_moves = self.generate_forward_moves()
            if not possible_moves: #jeśli nie ma ruchów do przodu
                possible_moves = self.generate_possible_moves()
            
        random_move = random.choice(possible_moves)
        self.do_move(random_move)
        self.player^=1
        
    def code_state(self, possible_moves):
        state = []
        
        #najpierd dodaje moje położenie
        my_poss = None
        for animal in range(1,9):
            if self.animal_poss[1][animal] != None:
                my_poss = self.animal_poss[1][animal]
        
        if my_poss == None: 
                print("didn't found agent poss")
                exit()
            
        state.append(my_poss[0])
        state.append(my_poss[1])
        
        #dodaje jak dleko moge iść w jakim kierunku
        up, down, left, right = 0, 0, 0, 0
        for move in possible_moves:
            if move.shift[0] >= 1: 
                down = move.shift[0] 
            elif move.shift[0] <= -1:
                up = move.shift[0]
            elif move.shift[1] >= 1:
                right = move.shift[1]
            elif move.shift[1] <= -1:
                left = move.shift[1]
                
        state.append(up)
        state.append(down)
        state.append(left)
        state.append(right)
        
        #dodaje pozycje przeciwnika
        for animal in range(1, 9):
            poss = self.animal_poss[0][animal]
            if poss == None:
                state.append(-1)
                state.append(-1)
            else:
                state.append(poss[0])
                state.append(poss[1])
        
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    def chose_action(self): #zwraca state, move
        global steps_done
        
        steps_done += 1
        #generuje możliwe ruchy
        possible_moves = self.generate_possible_moves()
        mapa = {
            0:None, #up
            1:None, #down
            2: None,#left 
            3: None #right
        }
            
        for move in possible_moves:
            if move.shift[0] >= 1: mapa[1] = move
            elif move.shift[0] <= -1: mapa[0] = move 
            elif move.shift[1] >= 1: mapa[3] = move 
            elif move.shift[1] <= -1: mapa[2] = move
        
        move = None
        state_tensor = self.code_state(possible_moves)
        
        with torch.no_grad():
            move_tensor = policy_net(state_tensor).max(1).indices.view(1, 1)
            move = move_tensor.item()

        
        # skojarzenie action z possible move
        # 0 -> up, 1-> down, 2-> left, 3-> right
        move = mapa[move]
                
        return state_tensor, move, move_tensor
    
    def make_smart_move(self):
        # wybieramy jaką akcje chcemy wykonać
        #state i next state to reprezentacja gry, action to move jeśli wykonałem niepoprawny action to action = None
        state, move, move_tensor = self.chose_action()
        next_state = None
        
        # wykonujemy tą akcje i sprawdzamy nagrode za nią
        if move == None:
            pass
        self.do_move(move)
        reward = torch.tensor([self.last_reward], device=device)
        
        # sprawdzenie next_state
        if move == None:
            next_state = None
            self.inapropriate_move = True
            
        else:
            possible_moves = self.generate_possible_moves()
            next_state = self.code_state(possible_moves)
        self.player^=1
        
        # dodanie ruchu do pamięci
        memory.push(state, move_tensor, next_state, reward)
        
        #optymalizacja modelu
        self.optimize_model()
        
    def aval_pawns_with_scales(self, p):
        ans = 0
        for animal in range(9):
            if self.animal_poss[p][animal] != None: ans += animal
        return ans

    def optimize_model(self):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()  
        
class DQN(nn.Module):
    def __init__(self, start_dim, action_dim):
        super(DQN, self).__init__()
        
        self.layer1 = nn.Linear(start_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_dim)
    
    def forward(self, env):
        env = F.relu(self.layer1(env))
        env = F.relu(self.layer2(env))
        env = self.layer3(env)
        return env
    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
################################################################################################ koniec implementacji funcji i klas
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#parametry do DQN
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 5e-3

OBSERVATION_DIM = 22
ACTION_DIM = 4

policy_net = DQN(OBSERVATION_DIM, ACTION_DIM).to(device)
target_net = DQN(OBSERVATION_DIM, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(5000)

steps_done = 0
sum_lenght, sum_kils, sum_deads = 0, 0, 0
vis = [[0] * 7 for _ in range(9)]

def main():
    animal_arg = sys.argv[1]
    trybe_arg = int(sys.argv[2])
    
    enum_of_animal = {
        "RAT": 1,
        "CAT": 2,
        "DOG": 3,
        "WOLF": 4, 
        "JAGUAR": 5,
        "TIGER": 6,
        "LION": 7, 
        "ELEPHANT": 8
    }
    
    animal_name = animal_arg
    animal_arg = enum_of_animal[animal_arg]
    
    path = f'bots/{animal_name}{trybe_arg}.pth' 
    policy_net.load_state_dict(torch.load(path, map_location=device))
    
    score = [0, 0]
    debug_cnt = 0
    for t in range(1000):
        board = [[pon(0, 2) for _ in range(7)] for _ in range(9)]
        animal_poss = [[None] * 9 for _ in range(2)]
        player = 0
        
        #ustawianie zwierząt dla random player 
        board[0][0].replace(7, 0); board[0][6].replace(6, 0); board[1][1].replace(3, 0); board[1][5].replace(2, 0); board[2][0].replace(1, 0); board[2][2].replace(5, 0); board[2][4].replace(4, 0); board[2][6].replace(8, 0)
        animal_poss[0][7] = (0, 0); animal_poss[0][6] = (0, 6); animal_poss[0][3] = (1, 1); animal_poss[0][2] = (1, 5); animal_poss[0][1] = (2, 0); animal_poss[0][5] = (2, 2); animal_poss[0][4] = (2, 4); animal_poss[0][8] = (2, 6)
        
        #ustawienie zwierząt dla smart player
        match animal_arg:
            case 1:
                board[6][6].replace(animal_arg, 1)
                animal_poss[1][animal_arg] = (6, 6)
            case 2:
                board[7][1].replace(animal_arg, 1)
                animal_poss[1][animal_arg] = (7, 1)
            case 3:
                board[7][5].replace(animal_arg, 1)
                animal_poss[1][animal_arg] = (7, 5)
            case 4:
                board[6][2].replace(animal_arg, 1)
                animal_poss[1][animal_arg] = (6, 2)
            case 5:
                board[6][4].replace(animal_arg, 1)
                animal_poss[1][animal_arg] = (6, 4)
            case 6:
                board[8][0].replace(animal_arg, 1)
                animal_poss[1][animal_arg] = (8, 0)
            case 7:
                board[8][6].replace(animal_arg, 1)
                animal_poss[1][animal_arg] = (8, 6)
            case 8:
                board[6][0].replace(animal_arg, 1)
                animal_poss[1][animal_arg] = (6, 0)
                
        #plansza ustawiona
        state = Board(player, board, animal_poss, trybe_arg)
        player_turn = player
        
        # P1 - góra - random - 0
        # P2 - dół - smart - 1
        # state.print_board()
        while not state.game_over():
            if player_turn == 0: #kolej randomowego gracza
                state.make_random_move()
            else: # kolej mądrego gracza
                state.make_smart_move()
                
            player_turn ^= 1
            # state.print_board()
            # print(f"rand score: {score[0]}, smart score: {score[1]}")
            
        print(f"round:{t} ||| smart score:{score[0]} ||| rand score:{score[1]} ||| ", end='')
        print(state.last_reward)
        
        if state.last_reward > 0:
            debug_cnt += 1
            
        # sprawdzenie czy random wygrał: 
        if state.board[8][3].animal or state.inapropriate_move or state.aval_pawns_with_scales(state.PSMART) == 0:
            score[1] += 1
        else:
            score[0] += 1
        
    #gry się zakończyly mogę zapisać model
    print(*score)
    print(sum_lenght, sum_kils, sum_deads)
    for i in range(9):
        print(*vis[i]) 
        
if __name__ == '__main__':
    main()