import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# specify the path to progress csv file.
root_dir = sys.argv[1]
plot_dir = os.path.join(root_dir, 'training_plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
csv_path = os.path.join(root_dir, 'progress.csv')
# csv_path = "swimmer_no_task_reward/demo/seed_1/progress.csv"

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
    # fig.savefig('plots_swimmer_no_task_rew/' + col + '.png')
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(col)))
    plt.close(fig)
