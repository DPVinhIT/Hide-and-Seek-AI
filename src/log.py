import argparse
import platform
import time
from datetime import datetime

from colorama import Fore, Style, init

from arena import Arena

# Khởi tạo colorama (hỗ trợ màu trên Windows)
init(autoreset=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple Pacman vs Ghost games"
    )
    parser.add_argument(
        "--seek", required=True, help="Student ID cho Pacman (Seeker)"
    )
    parser.add_argument(
        "--hide", required=True, help="Student ID cho Ghost (Hider)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Số trận muốn chạy (mặc định: 100)",
    )
    parser.add_argument(
        "--submissions-dir",
        default="../submissions",
        help="Thư mục chứa submissions",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Số bước tối đa mỗi trận"
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=3.0,
        help="Timeout mỗi bước (giây)",
    )
    args = parser.parse_args()

    # 🚫 Windows không hỗ trợ SIGALRM → bỏ timeout
    if platform.system() == "Windows":
        print(
            Fore.YELLOW
            + "⚠️  Windows detected → step timeout disabled.\n"
            + Style.RESET_ALL
        )
        args.step_timeout = None

    pacman_wins = 0
    ghost_wins = 0
    draws = 0
    errors = 0

    log_filename = f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(
        Fore.CYAN
        + f"\nBatch run started! Logs → {log_filename}\n"
        + Style.RESET_ALL
    )

    with open(log_filename, "w", encoding="utf-8") as log:
        for i in range(1, args.games + 1):
            try:
                arena = Arena(
                    pacman_id=args.seek,
                    ghost_id=args.hide,
                    submissions_dir=args.submissions_dir,
                    max_steps=args.max_steps,
                    visualize=False,
                    delay=0,
                    step_timeout=args.step_timeout,
                )

                arena.load_agents()
                result, _ = arena.run_game()

                if result == "pacman_wins":
                    pacman_wins += 1
                elif result == "ghost_wins":
                    ghost_wins += 1
                elif result == "draw":
                    draws += 1

                print(
                    Fore.YELLOW
                    + f"Progress: {i}/{args.games} games completed..."
                    + Style.RESET_ALL
                )
                log.write(f"Game {i}: {result}\n")
                log.flush()

            except Exception as e:
                errors += 1
                print(Fore.RED + f"⚠️  Error in game {i}: {e}" + Style.RESET_ALL)
                log.write(f"Error in game {i}: {e}\n")
                log.flush()

    # Tổng kết
    print(Fore.GREEN + "\nAll games completed!\n" + Style.RESET_ALL)
    print("=" * 50)
    print(Fore.CYAN + "STATISTICS SUMMARY".center(50) + Style.RESET_ALL)
    print("=" * 50)
    print(f"Total games : {args.games}")
    print(
        f"Pacman wins : {pacman_wins} ({pacman_wins / args.games * 100:.1f}%)"
    )
    print(f"Ghost wins  : {ghost_wins} ({ghost_wins / args.games * 100:.1f}%)")
    print(f"Draws       : {draws} ({draws / args.games * 100:.1f}%)")
    print(f"Errors      : {errors}")
    print("=" * 50)
    print(
        Fore.MAGENTA
        + f"\nDetailed log saved to: {log_filename}\n"
        + Style.RESET_ALL
    )


if __name__ == "__main__":
    main()
