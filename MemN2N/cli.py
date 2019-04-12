from collections import namedtuple

import click

from main import run


@click.command()
@click.option('--train', default=True, help="Train phase.")
@click.option('--file', default=False, help="Path of saved object file to load.")
@click.option('--local_atten', default=True, help="Dot connection mode with bias")
@click.option('--atten_dot_bias', default=False, help="Dot connection mode with bias")
@click.option('--atten_gen', default=False, help="General connection mode")
@click.option('--atten_con', default=False, help="Concat connection mode")
@click.option('--atten_per', default=False, help="Nonlinear connection mode")
@click.option('--nl_up', default=False, help="Nonlinear mapping to update of 'u' between hops")
@click.option('--save_dir', default='.save', help="Directory of saved object files.",
              show_default=True)
@click.option('--num_epochs', type=int, default=1000, help="The upper limit of epochs to train.",
              show_default=True)
@click.option('--batch_size', type=int, default=32, help="Batch size.", show_default=True)
@click.option('--lr', type=float, default=0.01, help="Learning rate.", show_default=True)
@click.option('--embed_size', type=int, default=128, help="Embedding size.",
              show_default=True)
@click.option('--task', type=int, default=3, help="Number of task to learn.",
              show_default=True)
@click.option('--memory_size', type=int, default=50, help="Capacity of memory.",
              show_default=True)
@click.option('--num_hops', type=int, default=3, help="Embedding size.", show_default=True)
@click.option('--max_clip', type=float, default=40.0, help="Max gradient norm to clip",
              show_default=True)
@click.option('--joint', is_flag=True, help="Joint learning.")
@click.option('--tenk', default=True, help="Use 10K dataset.")
@click.option('--use_bow', is_flag=False, help="Use BoW, or PE sentence representation.")
@click.option('--use_lw', default=True, help="Use layer-wise, or adjacent weight tying.")
@click.option('--use_ls', default=False, help="Use linear start.")
def cli(**kwargs):
    config = namedtuple("Config", kwargs.keys())(**kwargs)
    run(config)


if __name__ == "__main__":
    cli()
