# Generate song lyrics with fine-tuned GPT-2

## Instructions are for University of Tartu HPC Falcon

The [GPT-2 model](https://huggingface.co/gpt2) has been fine-tuned on the English-language part of 
[this dataset](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres) 
to generate lyrics given a song name.

Clone this repository:

`git clone https://github.com/lisskor/gpt2-lyrics-generation.git`

Copy model files into your directory 
(they are currently in `/tmp` on Rocket, I may need to find a more permanent way to do this):

`cp -r /tmp/gpt-2-lyrics ./gpt-2-lyrics`

Load Python module:

`module load any/python/3.8.3-conda`

Create and activate virtual environment:

```
conda create -n lyricsgen
conda activate lyricsgen
```

Install dependencies in the environment:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install transformers
```

Request a GPU from one of the Falcon clusters: 

`srun -t 01:00:00 -p gpu --gres=gpu:tesla:1 --pty bash`

After resources have been allocated, load Python there as well and activate the environment:

```
module load any/python/3.8.3-conda
conda activate lyricsgen
```

Run `python gpt2-lyrics-generation/generate_lyrics.py --help` to check that everything is working and see a description of available options.

I have had to reactivate the environment to make it work sometimes. 
If the script can't see `transformers`, check that the correct Python is used 
(it should be `~/.conda/envs/lyricsgen/bin/python`).

There are two ways to generate lyrics:

1. Interactively, where you provide a name for the song and generated lyrics are printed to stdout:
    
```python gpt2-lyrics-generation/generate_lyrics.py --model-dir gpt-2-lyrics --max-seq-length 150 --num-sequences 3 --reproducible```

You can control the length of each text in the output, 
the number of texts to generate for each song name, and whether to set the random seed or not.

2. From a text file containing one song name per line. The generated texts will be saved into `outputs.json`.

```python gpt2-lyrics-generation/generate_lyrics.py --input-names-file gpt2-lyrics-generation/example_names.txt --model-dir gpt-2-lyrics --max-seq-length 150 --num-sequences 3 --reproducible```
