import logging
import os
from math import ceil

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.option(
    '-o', '--out_path', type=click.Path(exists=False), default=None, show_default=True,
    help='output figure path, e.g.: results/history.png (If skipped the figure will be saved at the same directory of `csv_path`.)')
@click.option('--window', default=10, type=int, show_default=True, help='moving average window')
def visualize(csv_path, out_path, window):
    """
    CSV_PATH: input history csv path, e.g.: results/history.csv
    """
    _visualize(csv_path=csv_path, out_path=out_path, window=window)


def _visualize(csv_path, out_path=None, window=10):
    if out_path is None:
        out_path = os.path.splitext(csv_path)[0] + '.png'
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.set_index(df.columns[0], inplace=True)
    smoothed_df = df.rolling(window=window, center=True).mean()

    ncol = 2
    nrow = ceil(len(smoothed_df.columns) / ncol)
    axes = smoothed_df.plot(subplots=True, sharex=True, grid=True, figsize=(6 * ncol, 2 * nrow), layout=(nrow, ncol))

    axes.item(0).xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    fig = axes.item(0).figure
    fig.tight_layout()
    fig.savefig(out_path)
    logger.info(f'create figure at {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    visualize()
