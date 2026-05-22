from tkinter import Tk
from controllers.reversi_board_controller import ReversiBoardController
from controllers.reversi_board_controller import GAME_MODES
from model.reversi_model import ReversiGameModel
from ui.board_common import BoardCommonUI

__author__ = "danylofitel"


def get_reversi_model():
    return ReversiGameModel(difficulty=2)


def get_reversi_controller(player_moves_first):
    return ReversiBoardController(
        get_reversi_model(), GAME_MODES["playerVSPro"], player_moves_first
    )


def reversi(player_moves_first=True):
    root = Tk()
    controller = get_reversi_controller(player_moves_first)
    BoardCommonUI(root, controller)
    controller.fill_board()
    root.mainloop()


# Black moves first in Reversi. Set player_moves_first=False to play as White
# and let the AI play Black.
reversi(player_moves_first=True)
