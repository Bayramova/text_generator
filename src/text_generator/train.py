import click

import text_generator.data as data


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, dir_okay=False),
    default="data/data.txt",
    show_default=True,
    help="Path to file with data.",
)
def train(input_dir):
    """Script that trains a model and saves it to a file."""

    corpus = data.Corpus(input_dir)
