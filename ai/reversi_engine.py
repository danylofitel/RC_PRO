__author__ = "danylofitel"

from random import choice
from time import perf_counter


# Search bounds (treated as +/- infinity for negamax).
INT_MAX = 10**12
INT_MIN = -INT_MAX

# Heuristic weights.
MOBILITY_BONUS = 100
CORNER_CELL_BONUS = 100 * MOBILITY_BONUS
STABILITY_BONUS = CORNER_CELL_BONUS // 2
VICTORY_BONUS = INT_MAX // 4

# When True, get_best_move prints the chosen value and search depth.
DEBUG = True


# Reversi game engine. Board coordinates are (row, column), zero-indexed.
class ReversiEngine(object):
    # Cell contents / player identifiers.
    empty = 0
    first = 1
    second = 2

    # All eight (dx, dy) directions from a cell.
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    # Orthogonal directions from a corner along its two adjacent edges.
    edge_directions = ((-1, 0), (0, -1), (0, 1), (1, 0))

    # Sentinel returned by get_valid_moves / accepted by move() when no legal move exists.
    pass_move = (-1, -1)

    # -------- construction --------

    def __init__(self, board_size):
        self.size = board_size
        self.cell_count = board_size**2

        self.board = [
            [self.empty for _ in range(board_size)] for _ in range(board_size)
        ]

        # Standard Reversi starting position.
        mid = board_size // 2
        self.board[mid - 1][mid - 1] = self.second
        self.board[mid][mid] = self.second
        self.board[mid - 1][mid] = self.first
        self.board[mid][mid - 1] = self.first

        # score[i] = number of cells owned by player (i + 1).
        self.score = [2, 2]

        # Precomputed corner coordinates.
        m = board_size - 1
        self._corners = ((0, 0), (0, m), (m, 0), (m, m))

        # Free cells maintained incrementally by move()/undo_move().
        self._free_cells = {
            (x, y)
            for x in range(board_size)
            for y in range(board_size)
            if self.board[x][y] == self.empty
        }

        # Optional history of moves for callers that want undo without tracking
        # flipped_cells themselves. Populated only when move() is called with
        # push_stack=True; passes are not recorded.
        self.move_stack = []

        # Search statistics, (re)set at the start of each get_best_move call.
        # Initialised here so _negamax/_board_heuristics can also be called
        # standalone without raising AttributeError.
        self._search_depth = 0
        self._depth_reached = 0
        self._nodes_searched = 0
        self._cutoffs = 0

    def __repr__(self):
        header = "  " + " ".join(str(y) for y in range(self.size))
        rows = [
            "{0}|{1}|".format(
                x, " ".join(str(self.board[x][y]) for y in range(self.size))
            )
            for x in range(self.size)
        ]
        return header + "\n" + "\n".join(rows) + "\n"

    # -------- board geometry --------

    def is_on_board(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def is_edge(self, x, y):
        m = self.size - 1
        return x == 0 or x == m or y == 0 or y == m

    def is_corner(self, x, y):
        m = self.size - 1
        return (x == 0 or x == m) and (y == 0 or y == m)

    def get_corners(self):
        return self._corners

    # On-board neighbours of (x, y) in all 8 directions.
    def get_neighbours(self, x, y):
        return [
            (x + dx, y + dy)
            for dx, dy in self.directions
            if self.is_on_board(x + dx, y + dy)
        ]

    # -------- game state queries --------

    def get_opponent(self, player):
        if player == self.first:
            return self.second
        if player == self.second:
            return self.first
        raise ValueError("Invalid player: {0}".format(player))

    def is_over(self):
        # Board full, or one side wiped out: trivially over.
        if not self._free_cells or self.score[0] == 0 or self.score[1] == 0:
            return True
        # Otherwise over only if neither player has a legal placement.
        return self.get_valid_moves(self.first) == [
            self.pass_move
        ] and self.get_valid_moves(self.second) == [self.pass_move]

    # Returns the winning player, or self.empty (0) for a draw.
    def get_winner(self):
        if self.score[0] > self.score[1]:
            return self.first
        if self.score[0] < self.score[1]:
            return self.second
        return self.empty

    def get_score(self, player):
        return self.score[player - 1]

    def get_score_difference(self, player):
        return self.score[player - 1] - self.score[self.get_opponent(player) - 1]

    # -------- move generation / validation --------

    def get_free_cells(self):
        return list(self._free_cells)

    def get_valid_moves(self, player):
        moves = [
            cell
            for cell in self._free_cells
            if self.move_is_valid(player, cell[0], cell[1])
        ]
        return moves if moves else [self.pass_move]

    def move_is_valid(self, player, x, y):
        if not self.is_on_board(x, y) or self.board[x][y] != self.empty:
            return False
        return any(
            self._move_captures_direction(player, x, y, dx, dy)
            for dx, dy in self.directions
        )

    # True if placing `player` at (x, y) captures opponent pieces in direction (dx, dy).
    def _move_captures_direction(self, player, x, y, dx, dy):
        opponent = self.get_opponent(player)
        nx, ny = x + dx, y + dy

        # Must start by stepping onto an opponent cell.
        if not self.is_on_board(nx, ny) or self.board[nx][ny] != opponent:
            return False

        # Walk through the opponent's run; the move captures iff it ends on our own piece.
        while self.is_on_board(nx, ny) and self.board[nx][ny] == opponent:
            nx += dx
            ny += dy
        return self.is_on_board(nx, ny) and self.board[nx][ny] == player

    # -------- move execution --------

    def move(self, player, x, y, push_stack=False):
        # Pass: nothing to do, no cells flipped. Not recorded in the stack either.
        if (x, y) == self.pass_move:
            return []

        self.board[x][y] = player
        self.score[player - 1] += 1
        self._free_cells.discard((x, y))

        flipped = []
        for dx, dy in self.directions:
            flipped.extend(self._flip_cells_in_direction(player, x, y, dx, dy))

        if push_stack:
            self.move_stack.append((player, (x, y), flipped))
        return flipped

    # Flip opponent pieces in direction (dx, dy) and return the list of flipped coordinates.
    def _flip_cells_in_direction(self, player, x, y, dx, dy):
        if not self._move_captures_direction(player, x, y, dx, dy):
            return []

        opponent = self.get_opponent(player)
        flipped = []
        nx, ny = x + dx, y + dy
        while self.is_on_board(nx, ny) and self.board[nx][ny] == opponent:
            self.board[nx][ny] = player
            self.score[player - 1] += 1
            self.score[opponent - 1] -= 1
            flipped.append((nx, ny))
            nx += dx
            ny += dy
        return flipped

    def undo_move(self, player, x, y, flipped_cells, pop_stack=False):
        # Pass: there was no move to undo.
        if (x, y) == self.pass_move:
            return

        opponent = self.get_opponent(player)

        self.board[x][y] = self.empty
        self.score[player - 1] -= 1
        self._free_cells.add((x, y))

        for fx, fy in flipped_cells:
            self.board[fx][fy] = opponent
            self.score[player - 1] -= 1
            self.score[opponent - 1] += 1

        if pop_stack:
            self.move_stack.pop()

    # Undo the most recent move() that was called with push_stack=True. Does not
    # account for passes (passes were never pushed), so passing the turn between
    # two real moves and then calling undo_last_move() undoes the earlier real move.
    def undo_last_move(self):
        if not self.move_stack:
            raise IndexError("Moves stack is empty")
        player, (x, y), flipped = self.move_stack.pop()
        self.undo_move(player, x, y, flipped)

    # Compute the best move via negamax, play it, and return ((x, y), flipped_cells).
    def move_ai(self, player, difficulty):
        (x, y), _ = self.get_best_move(player, difficulty)
        return (x, y), self.move(player, x, y)

    # -------- heuristic evaluation --------
    # All midgame terms below are zero-sum: h(P, board) == -h(O, board).
    # Negamax depends on this; without it the AI sees inconsistent values at even
    # vs odd plies and plays incoherently.

    def _board_heuristics(self, player):
        if self.is_over():
            return self._victory_score_difference(player)
        return (
            self._mobility_score_difference(player)
            + self._corner_cells_score_difference(player)
            + self._stable_cells_score_difference(player)
        )

    # Number of legal placements for player. A forced pass counts as zero, so it
    # does not contribute to mobility.
    def _legal_move_count(self, player):
        moves = self.get_valid_moves(player)
        return 0 if moves == [self.pass_move] else len(moves)

    def _mobility_score_difference(self, player):
        opponent = self.get_opponent(player)
        return MOBILITY_BONUS * (
            self._legal_move_count(player) - self._legal_move_count(opponent)
        )

    def _corner_cells_score_difference(self, player):
        opponent = self.get_opponent(player)
        score = 0.0

        for cx, cy in self._corners:
            cell = self.board[cx][cy]
            if cell == player:
                score += 1
            elif cell == opponent:
                score -= 1
            else:
                # Empty corner: penalise whichever side has pieces next to it,
                # since they're one move away from giving up the corner.
                for nx, ny in self.get_neighbours(cx, cy):
                    nval = self.board[nx][ny]
                    if nval == player:
                        score -= 0.25
                    elif nval == opponent:
                        score += 0.25
        return int(CORNER_CELL_BONUS * score)

    def _stable_cells_score_difference(self, player):
        opponent = self.get_opponent(player)
        return STABILITY_BONUS * (
            len(self.get_stable_cells(player)) - len(self.get_stable_cells(opponent))
        )

    # Game-over reward. Magnitude dominates midgame terms by several orders of magnitude
    # so a guaranteed win/loss outranks any positional consideration. The per-cell margin
    # term breaks ties so a 64:0 sweep beats a 33:31 win.
    def _victory_score_difference(self, player):
        winner = self.get_winner()
        if winner == player:
            bonus = VICTORY_BONUS
        elif winner == self.get_opponent(player):
            bonus = -VICTORY_BONUS
        else:
            return 0
        return bonus + (
            VICTORY_BONUS // self.cell_count
        ) * self._final_score_difference(player)

    # Score difference assuming the game is over. Wipeouts are scored at +/- board size
    # rather than the actual remaining count, since the trailing side has "lost everything".
    def _final_score_difference(self, player):
        opponent = self.get_opponent(player)
        if self.score[player - 1] == 0:
            return -self.cell_count
        if self.score[opponent - 1] == 0:
            return self.cell_count
        return self.score[player - 1] - self.score[opponent - 1]

    # The raw counts behind the midgame terms, as (player, opponent) pairs:
    # legal moves (mobility), stable cells, and owned corners. Used for logging.
    def _cell_counts(self, player, opponent):
        mob_p = self._legal_move_count(player)
        mob_o = self._legal_move_count(opponent)
        stab_p = len(self.get_stable_cells(player))
        stab_o = len(self.get_stable_cells(opponent))
        corn_p = sum(self.board[cx][cy] == player for cx, cy in self._corners)
        corn_o = sum(self.board[cx][cy] == opponent for cx, cy in self._corners)
        return mob_p, mob_o, stab_p, stab_o, corn_p, corn_o

    # -------- stable cell detection --------

    # Pieces that can never be flipped. Two contributing patterns:
    #   1. An owned corner, plus any runs of the player's pieces extending outward
    #      from it along the two adjacent edges (until an enemy/empty/other corner).
    #   2. Any of the player's pieces lying on a fully-filled edge between corners.
    # Returns a set of (x, y).
    def get_stable_cells(self, player):
        stable = set()

        for cx, cy in self._corners:
            owns_corner = self.board[cx][cy] == player
            if owns_corner:
                stable.add((cx, cy))
            for dx, dy in self.edge_directions:
                # Runs extending from an owned corner along its adjacent edges.
                if owns_corner:
                    stable.update(
                        self._stable_cells_on_edges_in_direction_from_corner(
                            player, cx, cy, dx, dy
                        )
                    )
                # Pieces on a fully-filled edge anchored at this corner.
                stable.update(self._stable_cells_on_filled_edge(player, cx, cy, dx, dy))

        return stable

    # Walk outward from corner (x, y) in direction (dx, dy) and collect the player's
    # pieces until hitting another corner, an opponent/empty cell, or going off-board.
    # (x, y) itself is not included.
    def _stable_cells_on_edges_in_direction_from_corner(self, player, x, y, dx, dy):
        run = []
        nx, ny = x + dx, y + dy
        while (
            self.is_on_board(nx, ny)
            and self.board[nx][ny] == player
            and not self.is_corner(nx, ny)
        ):
            run.append((nx, ny))
            nx += dx
            ny += dy
        return run

    # If the edge starting at (x, y) in direction (dx, dy) is fully filled (no empties),
    # return all of the player's pieces on it. Otherwise return nothing.
    def _stable_cells_on_filled_edge(self, player, x, y, dx, dy):
        player_cells = []
        nx, ny = x, y
        while self.is_on_board(nx, ny):
            cell = self.board[nx][ny]
            if cell == self.empty:
                return []
            if cell == player:
                player_cells.append((nx, ny))
            nx += dx
            ny += dy
        return player_cells

    # -------- search --------

    # Pick the best move for the player using negamax. Ties broken uniformly at random.
    def get_best_move(self, player, difficulty):
        search_depth = self._compute_search_depth(difficulty)
        moves = self.get_valid_moves(player)
        opponent = self.get_opponent(player)

        self._nodes_searched = 0
        self._cutoffs = 0
        self._search_depth = search_depth
        self._depth_reached = 0
        t_start = perf_counter()

        # Forced pass: nothing to choose; still evaluate the resulting position.
        if moves == [self.pass_move]:
            value = -self._negamax(opponent, search_depth - 1, INT_MIN, INT_MAX)
            return self.pass_move, value

        # Full window at root so that equal-value ties can be detected exactly.
        # Internal alpha-beta inside _negamax still prunes within each subtree.
        best_value = INT_MIN
        best_moves = []
        for x, y in moves:
            flipped = self.move(player, x, y)
            value = -self._negamax(opponent, search_depth - 1, INT_MIN, INT_MAX)
            self.undo_move(player, x, y, flipped)

            if value > best_value:
                best_value = value
                best_moves = [(x, y)]
            elif value == best_value:
                best_moves.append((x, y))

        chosen = choice(best_moves)

        if DEBUG:
            elapsed = perf_counter() - t_start
            nps = int(self._nodes_searched / elapsed) if elapsed > 0 else 0
            # Counts for the position the chosen move creates (player/opponent).
            cx, cy = chosen
            flipped = self.move(player, cx, cy)
            mob_p, mob_o, stab_p, stab_o, corn_p, corn_o = self._cell_counts(
                player, opponent
            )
            self.undo_move(player, cx, cy, flipped)
            print(
                "value={0:>14} | mob={1:>2}/{2:<2} stab={3:>2}/{4:<2} "
                "corn={5}/{6} | dlim={7:>2} dreach={8:>2} | "
                "nodes={9:>9} cutoffs={10:>9} | time={11:>7.3f}s nps={12:>10}".format(
                    best_value,
                    mob_p,
                    mob_o,
                    stab_p,
                    stab_o,
                    corn_p,
                    corn_o,
                    search_depth,
                    self._depth_reached,
                    self._nodes_searched,
                    self._cutoffs,
                    elapsed,
                    nps,
                )
            )

        return chosen, best_value

    # Negamax with fail-soft alpha-beta pruning. Returns the value of the position
    # from `player`'s perspective (so the caller does `value = -_negamax(opp, ...)`).
    def _negamax(self, player, depth, alpha, beta):
        self._nodes_searched += 1

        if depth == 0 or self.is_over():
            depth_from_root = self._search_depth - depth
            if depth_from_root > self._depth_reached:
                self._depth_reached = depth_from_root
            return self._board_heuristics(player)

        moves = self.get_valid_moves(player)
        opponent = self.get_opponent(player)

        # Forced pass: hand the turn to the opponent without changing the board.
        if moves == [self.pass_move]:
            return -self._negamax(opponent, depth - 1, -beta, -alpha)

        best = INT_MIN
        for x, y in moves:
            flipped = self.move(player, x, y)
            value = -self._negamax(opponent, depth - 1, -beta, -alpha)
            self.undo_move(player, x, y, flipped)

            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._cutoffs += 1
                break  # fail-soft beta cutoff
        return best

    # Scale search depth with difficulty: deeper as the board fills up, and solve
    # exhaustively once the remaining tree is small enough.
    def _compute_search_depth(self, difficulty):
        search_depth = difficulty + 1
        filled = self.score[0] + self.score[1]
        empty = self.cell_count - filled

        if difficulty >= 2:
            # +1 ply per 10% of the board filled beyond 50%.
            if empty < filled:
                search_depth += int(10 * (filled / self.cell_count - 0.5))
            # Solve to the end of the game once the remaining tree is shallow enough.
            threshold = self.size + difficulty
            if search_depth <= empty and empty <= threshold:
                search_depth = empty + 1
        return search_depth
