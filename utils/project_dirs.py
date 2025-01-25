from pathlib import Path

def project_root():
    current_dir = Path(__file__).absolute().parent.parent
    return current_dir

def game_dir():
    return project_root() / "game"

def utils_dir():
    return project_root() / "utils"

def saved_df():
    res = project_root() / "saved_df"
    res.mkdir(parents=True, exist_ok=True)
    return res

def saved_df_n(n:int): #number of gammas 
    res = saved_df() / f"{n}_gamma"
    res.mkdir(parents=True, exist_ok=True)
    return res

def save_df_n_dist(n:int, distribution:str):
    res = saved_df_n(n = n) / distribution
    res.mkdir(parents=True, exist_ok=True)
    return res

if __name__ == "__main__":
    print(project_root())
    print(game_dir())
    print(utils_dir())