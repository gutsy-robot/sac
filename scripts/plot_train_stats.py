import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# specify the path to progress csv file.
csv_path = "swimmer_no_task_reward/demo/seed_1/progress.csv"

csv = pd.read_csv(csv_path)
for col in csv.columns:
    fig, ax = plt.subplots()
    fig.suptitle(col)
    plt.plot(csv[col].to_numpy())
    if '/' in col:
        splits = col.split('/')
        col = ""
        for s in splits:
            col += s
    # os.system('rm plots/' + col + '_swimmer.png')
    fig.savefig('plots_swimmer_no_task_rew/' + col + '.png')
    plt.close(fig)
