import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


# ========================
# PACMAN AGENT (Seeker)
# ========================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = []

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        start_time = time.time()

        # nếu còn đường đi thì đi tiếp
        if self.path:
            return self.path.pop(0)

        # BFS tìm đường ngắn nhất tới Ghost
        queue = [(my_position, [])]
        visited = {my_position}

        while queue:
            # quá 1 giây → Ghost thắng
            if time.time() - start_time > 1.0:
                raise TimeoutError("Pacman quá 1 giây → Ghost thắng")

            pos, path = queue.pop(0)
            if pos == enemy_position:
                if path:
                    self.path = path[1:]
                    return path[0]
                else:
                    return Move.STAY

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (pos[0] + dr, pos[1] + dc)
                if (
                    self._is_valid_position(new_pos, map_state)
                    and new_pos not in visited
                ):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [move]))

        # fallback nếu chưa tìm ra
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            new_pos = (my_position[0] + dr, my_position[1] + dc)
            if self._is_valid_position(new_pos, map_state):
                return move

        return Move.STAY

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        r, c = pos
        h, w = map_state.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return map_state[r, c] == 0


# ========================
# GHOST AGENT (Hider)
# ========================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Predictive + Minimax Ghost"
        self.prev_enemy = None

    def step(self, map_state, my_position, enemy_position, step_number):
        start_time = time.time()

        # ====== PREDICTIVE PHASE ======
        if self.prev_enemy:
            pac_dir = (
                enemy_position[0] - self.prev_enemy[0],
                enemy_position[1] - self.prev_enemy[1],
            )
        else:
            pac_dir = (0, 0)
        self.prev_enemy = enemy_position

        # Dự đoán vị trí pacman trong 2 bước tới
        predicted_pac = (
            enemy_position[0] + 2 * pac_dir[0],
            enemy_position[1] + 2 * pac_dir[1],
        )

        h, w = map_state.shape
        if not (0 <= predicted_pac[0] < h and 0 <= predicted_pac[1] < w):
            predicted_pac = enemy_position

        # ====== MINIMAX PHASE ======
        best_move = Move.STAY
        best_score = -999999

        valid_moves = [
            m
            for m in Move
            if self._is_valid(self._next(my_position, m), map_state)
        ]
        if not valid_moves:
            return Move.STAY

        for move in valid_moves:
            new_pos = self._next(my_position, move)
            score = self.minimax(
                map_state, new_pos, predicted_pac, depth=2, maximizing=False
            )
            if score > best_score:
                best_score = score
                best_move = move

        # tránh timeout
        if time.time() - start_time > 0.95:
            return Move.STAY
        return best_move

    # ====== MINIMAX ======
    def minimax(self, grid, ghost_pos, pac_pos, depth, maximizing):
        if depth == 0:
            return self._evaluate(ghost_pos, pac_pos, grid)

        moves = [
            m
            for m in Move
            if self._is_valid(
                self._next(ghost_pos if maximizing else pac_pos, m), grid
            )
        ]

        if not moves:
            return self._evaluate(ghost_pos, pac_pos, grid)

        if maximizing:
            best = -999999
            for m in moves:
                new_ghost = self._next(ghost_pos, m)
                val = self.minimax(grid, new_ghost, pac_pos, depth - 1, False)
                if val > best:
                    best = val
            return best
        else:
            best = 999999
            for m in moves:
                new_pac = self._next(pac_pos, m)
                val = self.minimax(grid, ghost_pos, new_pac, depth - 1, True)
                if val < best:
                    best = val
            return best

    # ====== ĐÁNH GIÁ HEURISTIC ======
    def _evaluate(self, ghost_pos, pac_pos, grid):
        # càng xa Pacman càng tốt
        dist = abs(ghost_pos[0] - pac_pos[0]) + abs(ghost_pos[1] - pac_pos[1])
        free = self._freedom(ghost_pos, grid)
        return dist + 0.3 * free

    def _freedom(self, pos, grid):
        drc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        count = 0
        for dr, dc in drc:
            nr, nc = pos[0] + dr, pos[1] + dc
            if self._is_valid((nr, nc), grid):
                count += 1
        return count

    def _next(self, pos, move):
        return (pos[0] + move.value[0], pos[1] + move.value[1])

    def _is_valid(self, pos, grid):
        r, c = pos
        h, w = grid.shape
        return 0 <= r < h and 0 <= c < w and grid[r, c] == 0
