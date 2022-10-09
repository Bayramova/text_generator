import click
import torch
import torch.nn as nn

import text_generator.data as data
from text_generator.model import LSTMModel


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, dir_okay=False),
    default="data/data.txt",
    show_default=True,
    help="Path to file with data.",
)
@click.option(
    "--batch-size", type=int, default=128, show_default=True, help="Batch size."
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
    "--nlayers", type=int, default=2, show_default=True, help="Number of layers."
)
@click.option(
    "--nepochs", type=int, default=20, show_default=True, help="Upper epoch limit."
)
@click.option(
    "--lr",
    type=float,
    default=0.0001,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--dropout",
    type=float,
    default=0.2,
    show_default=True,
    help="Dropout applied to layers (0 = no dropout).",
)
@click.option("--seed", type=int, default=1111, show_default=True, help="Random seed.")
@click.option(
    "--save",
    type=click.Path(),
    default="models/model.pt",
    show_default=True,
    help="Path to save the trained model.",
)
@click.option(
    "--log-interval",
    type=int,
    default=10,
    show_default=True,
    help="Report interval.",
)
def train(
    input_dir,
    batch_size,
    seq_len,
    nhid,
    nlayers,
    nepochs,
    seed,
    save,
    lr,
    dropout,
    log_interval,
):
    """Script that trains a model and saves it to a file."""

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Load data
    corpus = data.Corpus(input_dir)
    ntoken = len(corpus.dictionary)
    click.echo(f"Number of unique words in {input_dir}: {ntoken}")
    click.echo(f"Total number of words: {corpus.data.size(0)}")

    # Batchify data
    data_batchified = batchify(corpus.data, batch_size)

    # Pretrain word embeddings
    embedder = data.TextEmbedder()
    pretrained_weights = embedder(corpus.dictionary.idx2word)

    # Build the model
    model = LSTMModel(
        ntoken=ntoken,
        ninp=embedder.dim,
        nhid=nhid,
        nlayers=nlayers,
        dropout=dropout,
        weights=pretrained_weights,
    )

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, nepochs + 1):
        for batch, i in enumerate(range(0, data_batchified.size(0) - 1, seq_len)):
            input, targets = get_batch(data_batchified, i, seq_len)

            optimizer.zero_grad()

            output = model(input)
            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()

            if batch % log_interval == 0 and batch > 0:
                print(
                    f"| epoch {epoch} | {batch} / {len(data_batchified) // seq_len} batches \
                    | loss {loss.item(): .4f}"
                )

    # Save the model to file
    torch.save(model, save)
    click.echo(f"Model is saved to {save}")


def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts
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
