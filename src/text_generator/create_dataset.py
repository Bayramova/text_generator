from itertools import islice
import os
from pathlib import Path

import click
from corus import load_mokoron
from numpy.random import default_rng
from razdel import sentenize


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, dir_okay=False),
    default="data/db.sql",
    show_default=True,
    help="Path to file with data.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data",
    show_default=True,
    help="Path to folder to store created datasets.",
)
@click.option(
    "--size",
    type=click.IntRange(min=1, max=200000),
    default=100000,
    show_default=True,
    help="Number of records to include in dataset.",
)
@click.option(
    "--train-split",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.8,
    show_default=True,
    help="Proportion of the dataset to include in the train split.",
)
@click.option("--seed", type=int, default=1111, show_default=True, help="Random seed.")
def create_dataset(
    input_dir: Path, output_dir: Path, size: int, train_split: float, seed: int
) -> None:
    click.echo("Loading dataset...\n")
    records = load_mokoron(input_dir)

    sentences = []
    for record in islice(records, size):
        text = record.text
        text = text.replace("\n", " ")
        for sentence in sentenize(text):
            sentences.append(sentence.text)

    rng = default_rng(seed)
    rng.shuffle(sentences)

    TRAIN_SPLIT = int(len(sentences) * train_split)
    train_corpus = "\n".join(sentences[:TRAIN_SPLIT])
    valid_corpus = "\n".join(sentences[TRAIN_SPLIT:])

    click.echo(f"Number of sentences in train dataset: {TRAIN_SPLIT}")
    click.echo(
        f"Number of sentences in valid dataset: {len(sentences) - TRAIN_SPLIT}\n"
    )

    with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf8") as file:
        file.write(train_corpus)
    with open(os.path.join(output_dir, "valid.txt"), "w", encoding="utf8") as file:
        file.write(valid_corpus)
    click.echo(f"Datasets are saved to {output_dir}.")
