import numpy as np
import pandas as pd
import random as rd

N = 100000
FEATURES = 10

cols = "abcdefghijkmnopqrstuv"
columns = list(cols)[:FEATURES]

x = np.random.rand(N, FEATURES)

df = pd.DataFrame(x, columns = columns)
df["y"] = np.sin(df["a"].values) + np.cos(df["b"].values) + np.random.rand(N) * 0.001

df.to_csv("data.csv")


unary_funs = ["sinf", "cosf", "sqrtf"]
operators = ["+", "-"]

def random_program(depth=4):
    r = rd.randint(0,100)
    if depth == 0 or r < 30:
        c = rd.choice(columns)
        return f"_{c}_"
    elif r < 80:
        c = rd.choice(unary_funs)
        r = random_program(depth-1)
        return f"{c}({r})"
    else:
        c = rd.choice(operators)
        r1 = random_program(depth-1)
        r2 = random_program(depth-1)
        return f"({r1}) {c} ({r2})"


with open("functions.txt", "w") as f:
    for _ in range(1000):
        f.write(random_program() + "\n")

        


