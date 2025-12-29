import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

POS = [
    "I loved this movie, it was fantastic and inspiring.",
    "Great acting and an excellent story.",
    "Amazing film, really enjoyable from start to finish.",
    "Wonderful plot and brilliant performances.",
    "This was a beautiful and heartwarming movie.",
    "I highly recommend it, very entertaining.",
    "Superb direction and great soundtrack.",
    "One of the best movies I have seen.",
    "It was fun, emotional, and well made.",
    "A great experience, I would watch again."
]

NEG = [
    "I hated this movie, it was boring and terrible.",
    "Worst acting I have ever seen.",
    "The story was awful and made no sense.",
    "A complete waste of time, very disappointing.",
    "Terrible film, I regret watching it.",
    "Poor directing and bad dialogue.",
    "It was slow, dull, and annoying.",
    "One of the worst movies ever.",
    "Unwatchable and badly made.",
    "Not recommended, it was painful to finish."
]

def write_samples(base_dir: str, label: str, samples: list[str], repeat: int = 4):
    folder = os.path.join(base_dir, label)
    os.makedirs(folder, exist_ok=True)
    idx = 1
    for _ in range(repeat):
        for s in samples:
            with open(os.path.join(folder, f"{idx}.txt"), "w", encoding="utf-8") as f:
                f.write(s)
            idx += 1

if __name__ == "__main__":
    base = DATA_DIR

    write_samples(base, "pos", POS, repeat=4)
    write_samples(base, "neg", NEG, repeat=4)
    print("Generated dataset in ./data (pos/neg)")
