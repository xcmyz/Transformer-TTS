import os
from multiprocessing import cpu_count
from tqdm import tqdm
from data import ljspeech


def preprocess_ljspeech(filename):
    in_dir = filename
    out_dir = "dataset"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')


def main():
    path = os.path.join("data", "LJSpeech-1.1")
    preprocess_ljspeech(path)


if __name__ == "__main__":
    main()
