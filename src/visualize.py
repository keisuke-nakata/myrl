import logging
import logging.config
import os

import click
import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.argument('out_path', type=click.Path(exists=False))
@click.option('--window', default=10, type=int, show_default=True, help='moving average window')
def visualize(csv_path, out_path, window):
    """
    CSV_PATH: input history csv path, e.g.: results/history.csv\n
    OUT_PATH: output figure path, e.g.: results/history.png
    """
    _visualize(csv_path=csv_path, out_path=out_path, window=window)


def _visualize(csv_path, out_path, window=10):
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col='total_episodes')
    smoothed_df = df.rolling(window=window, center=True).mean()
    axes = smoothed_df.plot(subplots=True, sharex=True, grid=True, figsize=(8, 12))
    fig = axes[0].figure
    fig.savefig(out_path)
    logger.info(f'create figure at {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    visualize()
