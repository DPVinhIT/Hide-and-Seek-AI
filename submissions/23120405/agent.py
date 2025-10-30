"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- You MUST return a Move enum value from step()
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
"""

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import time

import numpy as np

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost

    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        # - self.name = "Your Agent Name"
        pass

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here

        # Example: Simple greedy approach (replace with your algorithm)
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]

        # Try to move towards ghost
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT

        # Check if move is valid
        if self._is_valid_move(my_position, move, map_state):
            return move

        # If not valid, try other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move

        return Move.STAY

    # Helper methods (you can add more)

    def _is_valid_move(
        self, pos: tuple, move: Move, map_state: np.ndarray
    ) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0


# ========================
# GHOST AGENT (Hider)
# ========================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hide and Seek AI"
        # Lưu vị trí Pacman ở bước trước để suy ra hướng di chuyển gần nhất
        self.prev_enemy = None

    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Hàm chính được gọi mỗi bước.
        Trả về một Move (UP/DOWN/LEFT/RIGHT/STAY).
        """

        # 1) Ghi thời gian bắt đầu để tránh tính toán quá lâu (timeout guard)
        start_time = time.time()

        # -----------------------
        # PREDICTIVE PHASE
        # -----------------------
        # 2) Nếu có vị trí Pacman ở bước trước, suy ra vector di chuyển gần nhất (pac_dir)
        #    pac_dir = (delta_row, delta_col). Nếu chưa có, mặc định (0,0).
        if self.prev_enemy:
            pac_dir = (
                enemy_position[0] - self.prev_enemy[0],
                enemy_position[1] - self.prev_enemy[1],
            )
        else:
            pac_dir = (0, 0)

        # 3) Cập nhật prev_enemy để lần sau có thể tính pac_dir
        self.prev_enemy = enemy_position

        # 4) Dự đoán vị trí Pacman sau 2 bước: predicted_pac = current + 2 * pac_dir
        predicted_pac = (
            enemy_position[0] + 2 * pac_dir[0],
            enemy_position[1] + 2 * pac_dir[1],
        )

        # 5) Nếu vị trí dự đoán nằm ngoài bản đồ thì dùng vị trí hiện tại của Pacman
        h, w = map_state.shape
        if not (0 <= predicted_pac[0] < h and 0 <= predicted_pac[1] < w):
            predicted_pac = enemy_position

        # -----------------------
        # MINIMAX + ALPHA-BETA PHASE
        # -----------------------
        # 6) Chuẩn bị biến lưu best move và best score (điểm càng lớn càng tốt với Ghost)
        best_move = Move.STAY
        best_score = float("-inf")

        # 7) Lấy danh sách các move hợp lệ từ vị trí hiện tại
        #    _is_valid_move kiểm tra trong bounds & không phải wall.
        valid_moves = [
            m for m in Move if self._is_valid_move(my_position, m, map_state)
        ]
        # 8) Nếu không có move hợp lệ (bị kẹt) → đứng yên
        if not valid_moves:
            return Move.STAY

        # 9) Duyệt từng move hợp lệ:
        #    - tính vị trí mới new_pos nếu ghost đi move đó
        #    - mô phỏng minimax (với alpha-beta pruning) bắt đầu từ trạng thái giả lập
        #      + ghost đã đi new_pos
        #      + pacman được đặt tại predicted_pac (dự đoán)
        #      + depth nhỏ (ví dụ 2) để tránh tốn thời gian
        for move in valid_moves:
            new_pos = self._next_position(my_position, move)

            score = self.minimax(
                map_state,
                new_pos,
                predicted_pac,
                depth=2,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=False,
            )

            # 10) Cập nhật move tốt nhất nếu score lớn hơn
            if score > best_score:
                best_score = score
                best_move = move

            # 11) Kiểm tra thời gian: nếu gần timeout thì dừng sớm, trả move tốt nhất hiện có (1 step/ 1s)
            if time.time() - start_time > 0.95:
                break

        # 12) Trả về move tốt nhất tìm được
        return best_move

    # ====== MINIMAX + ALPHABETA ======
    def minimax(self, grid, ghost_pos, pac_pos, depth, alpha, beta, maximizing):
        """
        Hàm minimax với alpha-beta pruning.
        - grid: bản đồ (numpy array)
        - ghost_pos: vị trí ghost trong nhánh giả lập
        - pac_pos: vị trí pacman trong nhánh giả lập
        - depth: độ sâu còn lại
        - alpha, beta: bounds để pruning
        - maximizing: True khi đến lượt Ghost (maximizer); False khi lượt Pacman (minimizer)
        Trả về giá trị heuristic (score) cho trạng thái.
        """

        # 13) Base case: nếu depth == 0 -> trả về đánh giá heuristic của trạng thái
        if depth == 0:
            return self._evaluate(ghost_pos, pac_pos, grid)

        # 14) Lấy danh sách move hợp lệ cho người đang đi:
        #     - nếu maximizing thì lấy move hợp lệ từ ghost_pos
        #     - nếu minimizing thì lấy move hợp lệ từ pac_pos
        moves = [
            m
            for m in Move
            if self._is_valid_move(
                ghost_pos if maximizing else pac_pos, m, grid
            )
        ]

        # 15) Nếu không có move (bị kẹt) -> trả heuristic trực tiếp
        if not moves:
            return self._evaluate(ghost_pos, pac_pos, grid)

        # 16) Nếu là lượt Ghost (maximizer): cố gắng tối đa hóa score
        if maximizing:
            value = float("-inf")
            for m in moves:
                new_ghost = self._next_position(ghost_pos, m)
                # 16.1) Gọi đệ quy cho lượt tiếp theo (Pacman), giảm depth
                val = self.minimax(
                    grid, new_ghost, pac_pos, depth - 1, alpha, beta, False
                )
                # 16.2) giữ giá trị lớn nhất
                if val > value:
                    value = val
                # 16.3) cập nhật alpha và cắt nhánh khi alpha >= beta
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
            return value

        # 17) Nếu là lượt Pacman (minimizer): cố gắng giảm score Ghost
        else:
            value = float("inf")
            for m in moves:
                new_pac = self._next_position(pac_pos, m)
                # 17.1) Gọi đệ quy cho lượt tiếp theo (Ghost)
                val = self.minimax(
                    grid, ghost_pos, new_pac, depth - 1, alpha, beta, True
                )
                # 17.2) giữ giá trị nhỏ nhất
                if val < value:
                    value = val
                # 17.3) cập nhật beta và cắt nhánh khi alpha >= beta
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break
            return value

    # ====== HEURISTIC / HÀM ĐÁNH GIÁ ======
    def _evaluate(self, ghost_pos, pac_pos, grid):
        """
        Hàm đánh giá trạng thái:
        - mục tiêu Ghost: score càng lớn càng tốt (xa Pacman + có nhiều lối thoát)
        - dist: khoảng cách Manhattan giữa Ghost và Pacman (càng lớn càng tốt)
        - free: số ô trống quanh Ghost (ưu tiên vị trí có lối thoát)
        - trả về dist + 0.3 * free (hệ số 0.3 có thể tinh chỉnh)
        """
        dist = abs(ghost_pos[0] - pac_pos[0]) + abs(ghost_pos[1] - pac_pos[1])
        free = self._freedom(ghost_pos, grid)
        return dist + 0.3 * free

    def _freedom(self, pos, grid):
        """
        Đếm số ô trống (UP/DOWN/LEFT/RIGHT) quanh pos.
        Mục đích: ưu tiên những ô có nhiều lựa chọn thoát.
        """
        directions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        count = 0
        for d in directions:
            new_p = self._next_position(pos, d)
            if self._is_valid_position(new_p, grid):
                count += 1
        return count

    # ====== HÀM TIỆN ÍCH ======
    def _next_position(self, pos, move):
        """Trả về vị trí mới = pos + move.value (không kiểm tra hợp lệ ở đây)."""
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _is_valid_move(
        self, pos: tuple, move: Move, map_state: np.ndarray
    ) -> bool:
        """
        Kiểm tra move có hợp lệ không:
        - tính new_pos = pos + move
        - kiểm tra new_pos nằm trong bounds và grid[new_pos] == 0 (không phải wall)
        """
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """
        Kiểm tra new_pos nằm trong bản đồ và là ô trống (0), trả về True/False.
        """
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0
