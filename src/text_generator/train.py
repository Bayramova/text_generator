import click

import text_generator.data as data
from text_generator.model import RNNModel


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, dir_okay=False),
    default="data/data.txt",
    show_default=True,
    help="Path to file with data.",
)
@click.option(
    "--batch-size", type=int, default=16, show_default=True, help="Batch size."
)
@click.option(
    "--seq-len", type=int, default=10, show_default=True, help="Sequence length."
)
@click.option(
    "--nhid",
    type=int,
    default=300,
    show_default=True,
    help="Number of hidden units per layer.",
)
@click.option(
    "--nlayers", type=int, default=1, show_default=True, help="Number of layers."
)
def train(input_dir, batch_size, seq_len, nhid, nlayers):
    """Script that trains a model and saves it to a file."""
    # Load data
    corpus = data.Corpus(input_dir)
    ntoken = len(corpus.dictionary)
    click.echo(f"Number of unique words in {input_dir}: {ntoken}")

    # Batchify data
    data_batchified = batchify(corpus.data, batch_size)

    # Pretrain word embeddings
    embedder = data.TextEmbedder()
    pretrained_weights = embedder(corpus.dictionary.idx2word)

    # Build the model
    model = RNNModel(
        ntoken=ntoken,
        ninp=embedder.dim,
        nhid=nhid,
        nlayers=nlayers,
        weights=pretrained_weights,
    )


def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders)
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target
