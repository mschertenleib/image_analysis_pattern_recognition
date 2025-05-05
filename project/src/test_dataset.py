import argparse

from dataset import *


def main(args: argparse.Namespace) -> None:
    dataset = ReferenceDataset(args.dir, args.contours_file, device="cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.path.join("data", "project", "references"))
    parser.add_argument(
        "--contours_file", type=str, default=os.path.join("project", "src", "contours.json")
    )
    args = parser.parse_args()

    main(args)
