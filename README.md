# Text Generator
This project allows you to generate wolf/patsanskie quotes.

## Usage
1. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Python 3.9.7 and Poetry 1.1.13).
2. Clone this repository to your machine.
3. Install project dependencies (run this and following commands in a terminal, from the root of cloned repository):
```
poetry install --no-dev
```
4. Run train with the following command:
```
poetry run train --input-dir <path to txt with data> --save <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```
poetry run train --help
```
5. Run generate with the following command:
```
poetry run generate --input-dir <path to txt with data> --checkpoint <path to trained model> --prefix <seed word> --length <number of words to generate>
```

## Development

The code in this repository must be linted with flake8 and formatted with black before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer tools.

Format your code with [black](https://github.com/psf/black) and lint it with [flake8](https://github.com/PyCQA/flake8):
```
poetry run black src
```
```
poetry run flake8 src
```
