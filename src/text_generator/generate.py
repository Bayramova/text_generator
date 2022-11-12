import click
import torch

import text_generator.data as data


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, dir_okay=True),
    default="data",
    show_default=True,
    help="Path to folder with data.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/generated.txt",
    show_default=True,
    help="Path to file to store generated sequences.",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default="models/model.pt",
    show_default=True,
    help="Path to the trained model.",
)
@click.option(
    "--prefix",
    type=str,
    default=None,
    show_default=True,
    help="Seed word (randomly selected from the dictionary, if not given or doesn't exist).",
)
@click.option(
    "--length",
    type=int,
    default=7,
    show_default=True,
    help="Number of words to generate.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    show_default=True,
    help="Temperature - higher will increase diversity.",
)
def generate(input_dir, output_dir, checkpoint, prefix, length, temperature):
    """Generates new sentences sampled from the language model."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    model = torch.load(checkpoint)

    # Load data
    corpus = data.Corpus(input_dir)
    ntoken = len(corpus.dictionary)

    if prefix is None or prefix not in corpus.dictionary.word2idx:
        # Choose random word from dictionary
        input = torch.randint(ntoken, (1, 1), dtype=torch.int64).to(device)
        # click.echo(f"Random prefix: {corpus.dictionary.idx2word[input]}\n")
    else:
        input = (
            torch.tensor(corpus.dictionary.word2idx[prefix]).reshape((1, 1)).to(device)
        )

    generated_sequence = [corpus.dictionary.idx2word[input]]
    for _ in range(length):
        output = model(input)
        word_weights = output.div(temperature)
        word_weights = torch.nn.functional.softmax(word_weights, dim=1)
        word_idx = torch.multinomial(word_weights, 1)
        input.fill_(word_idx.item())

        word = corpus.dictionary.idx2word[word_idx]
        generated_sequence.append(word)

    generated_sequence = " ".join(generated_sequence)
    click.echo(f"Generated sequence:\n{generated_sequence}\n")

    # Save generated sequence to file
    with open(output_dir, "a", encoding="utf8") as file:
        file.write(f"{generated_sequence}\n")
    click.echo(f"Generated sequence is saved to {output_dir}")
