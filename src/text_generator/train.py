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
    data_batchified = batchify(corpus.data)


def batchify(data, batch_size=4):
    # Work out how cleanly we can divide the dataset into bsz parts
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders)
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(source, i):
    seq_len = min(10, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target
