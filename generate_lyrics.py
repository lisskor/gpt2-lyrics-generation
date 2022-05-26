import logging
import json
from argparse import ArgumentParser
from transformers import pipeline, set_seed

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def load_model(args):
    # load the model
    logging.info("Loading model...")
    gpt_generator = pipeline('text-generation', model=args.model_dir)
    logging.info(f"Loaded model from {args.model_dir}")
    # set random seed, if using
    if args.reproducible:
        set_seed(42)
    return gpt_generator


def generate_lyrics_interactive(gpt_generator, args):
    # ask for user input in a loop
    song_name = ''
    while song_name != 'quit':
        song_name = input("Type a name for a song, or enter 'quit' to exit: ").strip()
        if song_name != 'quit':
            prompt = f"Lyrics to {song_name} ::: "
            for i, seq in enumerate(gpt_generator(prompt, max_length=args.max_seq_length,
                                                  num_return_sequences=args.num_sequences)):
                print(f"Text {i+1}:")
                print(prompt[:-4])
                print(seq['generated_text'][len(prompt):] + '\n')


def generate_lyrics_from_file(gpt_generator, args):
    with open(args.input_names_file, 'r', encoding='utf-8') as in_fh:
        song_names = [line.strip() for line in in_fh.readlines()]
    with open('outputs.json', 'w', encoding='utf-8') as out_fh:
        for name in song_names:
            prompt = f"Lyrics to {name} ::: "
            for i, seq in enumerate(gpt_generator(prompt, max_length=args.max_seq_length,
                                                  num_return_sequences=args.num_sequences)):
                json_dict = json.dumps({'name': f"{name}_{i}", "lyrics": seq['generated_text'][len(prompt):]})
                out_fh.write(json_dict + '\n')
    logging.info("Saved lyrics into outputs.json")


if __name__ == '__main__':
    parser = ArgumentParser(description="""Generate text with a GPT2-type model""")
    parser.add_argument("--model-dir", required=True, type=str,
                        help="Path to model location")
    parser.add_argument("--max-seq-length", type=int, default=150,
                        help="""Number of tokens to generate in each sequence 
                                (tokens do not exactly equal words, there may be fewer words than tokens)""")
    parser.add_argument("--num-sequences", type=int, default=3,
                        help="Number of sequences to return")
    parser.add_argument("--reproducible", action="store_true",
                        help="Set random seed to obtain reproducible results (otherwise there is some randomness)")
    parser.add_argument("--input-names-file", type=str,
                        help="""If given, instead of interactive generation, will read song names from a file
                                (one name per line) and save output texts into outputs.json""")

    arguments = parser.parse_args()

    generator = load_model(arguments)

    if arguments.input_names_file:
        generate_lyrics_from_file(generator, arguments)
    else:
        generate_lyrics_interactive(generator, arguments)

    logging.info("Exiting")
