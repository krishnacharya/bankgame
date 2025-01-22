from pathlib import Path

def project_root():
    current_dir = Path(__file__).absolute().parent.parent
    return current_dir

def game_dir():
    return project_root() / "game"

def utils_dir():
    return project_root() / "utils"

if __name__ == "__main__":
    print(project_root())
    print(game_dir())
    print(utils_dir())