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

# def saved_df_2gamma(epsigns:str, n = 2):
#     '''
#     epsigns is one of ['sign++','sign+-, sign-+, sign--'] # denotes sign for epsilon_1 and epsilon_2
    
#     instance_name: has the distribution name and the gamma array concatenated as string, 
#     for e.g.  Truncg_mu{mu}_sig{sigma}_gamma[0.4,0.6], Punif_gamma[0.4, 0.6]
#     '''
#     res = saved_df_n(n=n) / epsigns
#     res.mkdir(parents=True, exist_ok=True)
#     return res

def saved_df_n_dist(n:int, distribution:str):
    res = saved_df_n(n = n) / distribution
    res.mkdir(parents=True, exist_ok=True)
    return res

def saved_df_n_dist_concise(n:int, distribution:str):
    res = saved_df_n_dist(n, distribution) / "concise"
    res.mkdir(parents=True, exist_ok=True)
    return res

def saved_df_n_dist_full(n:int, distribution:str):
    res = saved_df_n_dist(n, distribution) / "full"
    res.mkdir(parents=True, exist_ok=True)
    return res

if __name__ == "__main__":
    print(project_root())
    print(game_dir())
    print(utils_dir())