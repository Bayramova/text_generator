from itertools import islice

import click
from corus import load_mokoron
from razdel import sentenize

path = "data/db.sql"


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
    default="data/data.txt",
    show_default=True,
    help="Path to file to store created dataset.",
)
@click.option(
    "--size",
    type=click.IntRange(min=1, max=200000),
    default=100000,
    show_default=True,
    help="Number of sentences to include in dataset.",
)
def create_dataset(input_dir, output_dir, size):
    records = load_mokoron(input_dir)
    sentences = []
    for record in islice(records, size):
        text = record.text
        text = text.replace("\n", " ")
        for sentence in sentenize(text):
            sentences.append(sentence.text)
    corpus = "\n".join(sentences)
    with open(output_dir, "w", encoding="utf8") as file:
        file.write(corpus)
