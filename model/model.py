__author__ = 'danylo'

BOARD_SIZE = 64


class GameModel:

    def create_board(self):
        #TODO remove it later
        board_position = [0 for i in range(BOARD_SIZE)]
        board_position[28] = 1
        board_position[27] = 1
        board_position[36] = 1
        board_position[35] = 1
        self.board_position = board_position

    def __init__(self):
        self.winner = None
        self.is_game_over = False
        self.current_player = 1
        self.board_position = None
        self.create_board()
        self.current_state = 0

    def is_game_over(self):
        return self.is_game_over

    def winner(self):
        return self.winner

    def get_current_player(self):
        pass

    def get_available_moves(self):
        pass
