import json
import random

def check_if_board_right(board_json):

    try:
        # Vérifier que le JSON est un dictionnaire
        if not isinstance(board_json, dict):
            return False, "Le JSON doit être un dictionnaire."
        
        # Vérifier la présence des clés nécessaires
        required_keys = {"rows", "columns", "board", "currentPlayer"}
        if not required_keys.issubset(board_json.keys()):
            return False, f"Clés manquantes : {required_keys - board_json.keys()}"

        # Vérifier les dimensions
        rows = board_json["rows"]
        columns = board_json["columns"]
        board = board_json["board"]
        
        if not (isinstance(rows, int) and rows > 0):
            return False, "La clé 'rows' doit être un entier positif."
        if not (isinstance(columns, int) and columns > 0):
            return False, "La clé 'columns' doit être un entier positif."

        if not (isinstance(board, list) and len(board) == rows):
            return False, "Le plateau doit être une liste de longueur égale à 'rows'."
        
        for row in board:
            if not (isinstance(row, list) and len(row) == columns):
                return False, f"Toutes les lignes doivent avoir une longueur de {columns}."
            if not all(cell in {0, 1, 2} for cell in row):
                return False, "Les valeurs des cases doivent être 0, 1 ou 2."

        # Vérifier la valeur de 'currentPlayer'
        current_player = board_json["currentPlayer"]
        if current_player not in {1, 2}:
            return False, "La clé 'currentPlayer' doit valoir 1 ou 2."

        # Si toutes les vérifications passent
        return True, "Le format du plateau est valide."

    except Exception as e:
        return False, f"Erreur lors de la validation : {str(e)}"


def play_random(json_info):

    is_valid, message = check_if_board_right(json_info)
    if not is_valid:
        raise ValueError(message)

    board = json_info["board"]
    player = json_info["currentPlayer"]

    # Identifier les colonnes où un coup est possible
    colonnes_valides = [col for col in range(len(board[0])) if board[0][col] == 0]

    # Si aucune colonne valide, retourner le board inchangé
    if not colonnes_valides:
        raise ValueError("Aucun coup possible")

    # Choisir une colonne valide au hasard
    colonne = random.choice(colonnes_valides)

    # Placer le jeton dans la colonne choisie
    for ligne in range(len(board) - 1, -1, -1):  # On parcourt les lignes de bas en haut
        if board[ligne][colonne] == 0:
            board[ligne][colonne] = player
            break

    # Créer un dictionnaire pour le coup joué
    coup = {
        "colonne": colonne,
        "ligne": ligne,
        "player": player
    }

    # Retourner le coup joué en format JSON
    return json.dumps(coup)


def test():

    board = {
        "rows": 6,
        "columns": 7,
        "board": [
            [0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 2, 1, 0, 0, 0], 
            [0, 1, 2, 1, 2, 0, 0]
        ],
        "currentPlayer": 1
    }

    print(f"JSON envoyé : {board}")

    move = play_random(board)

    print(f"JSON retourné : {move}")

    return move

if __name__ == "__main__":
    test()