import numpy as np
import torch
import math
import json

import torch.nn as nn
import torch.nn.functional as F

from flask import Flask, render_template, request, jsonify





class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value 
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
        
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

def check_if_board_right(board):
    
    # Vérification de la clé "board" : doit être une liste de 42 éléments (6 x 7)
    if not isinstance(board.get("board"), list) or len(board["board"]) != 42:
        return False, "Invalid board size"
    
    # Vérification que chaque élément du "board" est soit 0, 1, soit -1
    for cell in board["board"]:
        if cell not in [0, 1, -1]:
            return False, "Invalid cell value in board"
    
    # Vérification de la clé "currentPlayer" : doit être 1 ou -1
    if board.get("currentPlayer") not in [-1, 1]:
        return False, "Invalid current player"
    
    return True, "Board is valid"


def check_victory(board):
    rows = len(board)
    cols = len(board[0])
    
    # Vérifier les alignements horizontaux
    for i in range(rows):
        for j in range(cols - 3):  # On s'arrête à 4 colonnes avant la fin
            if board[i][j] != 0 and board[i][j] == board[i][j + 1] == board[i][j + 2] == board[i][j + 3]:
                return True, board[i][j]
    
    # Vérifier les alignements verticaux
    for i in range(rows - 3):  # On s'arrête à 3 lignes avant la fin
        for j in range(cols):
            if board[i][j] != 0 and board[i][j] == board[i + 1][j] == board[i + 2][j] == board[i + 3][j]:
                return True, board[i][j]
    
    # Vérifier les diagonales descendantes
    for i in range(rows - 3):
        for j in range(cols - 3):
            if board[i][j] != 0 and board[i][j] == board[i + 1][j + 1] == board[i + 2][j + 2] == board[i + 3][j + 3]:
                return True, board[i][j]
    
    # Vérifier les diagonales montantes
    for i in range(3, rows):
        for j in range(cols - 3):
            if board[i][j] != 0 and board[i][j] == board[i - 1][j + 1] == board[i - 2][j + 2] == board[i - 3][j + 3]:
                return True, board[i][j]
    
    # Si aucune condition de victoire n'est trouvée
    return False, 0


game = ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("ia/model_7_ConnectFour.pt", map_location=device))
model.eval()

def play_mcts(json_info):

    print(json_info)

    # is_valid, message = check_if_board_right(json_info)
    # if not is_valid:
    #     raise ValueError(message)

    # Convertion du board de liste vers matrice
    board = json_info
    board =  [board[i:i + 7] for i in range(0, len(board), 7)]

    is_human_win, winner = check_victory(board)

    if is_human_win == True and winner == 1:
        # Return Victoire Humain
        info_json = {
            "coup": -1,
            "player_win": winner
        }
        return info_json

    player = -1

    # Identifier les colonnes où un coup est possible
    colonnes_valides = [col for col in range(len(board[0])) if board[0][col] == 0]

    # Si aucune colonne valide, retourner le board inchangé
    if not colonnes_valides:
        raise ValueError("Aucun coup possible")

    args = {
        'C': 2,
        'num_searches': 60,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }

    mcts = MCTS(game, args, model)

    # Convertion du board en np.array
    state = np.array(board)

    neutral_state = game.change_perspective(state, player)

    mcts_probs = mcts.search(neutral_state)
    action = int(np.argmax(mcts_probs))

    # Placer le jeton dans la colonne choisie
    for ligne in range(len(board) - 1, -1, -1):  # On parcourt les lignes de bas en haut
        if board[ligne][action] == 0:
            board[ligne][action] = player
            break

    is_human_ia, winner = check_victory(board)

    if is_human_ia == True and winner == -1:
        # Return Victoire Humain
        info_json = {
            "coup": action + ligne*7,
            "player_win": winner
        }
        return info_json

    info_json = {
        "coup": action + ligne*7,
        "player_win": 0
    }
    return info_json
    return json.dumps(coup)



app = Flask(__name__)

# Route pour afficher la page HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route pour gérer le formulaire et utiliser la fonction Python
@app.route('/resultat', methods=['POST'])
def resultat():
    valeur = request.json.get('valeur')  # Récupère les données envoyées par l'utilisateur
    resultat = play_mcts(valeur)  # Appelle la fonction Python
    print(resultat)
    return resultat

if __name__ == '__main__':
    app.run(debug=True)