import argparse

from bertvectoriser import BertVectoriser


def main():
    parser = argparse.ArgumentParser()
    x = BertVectoriser(preload=False)

    args = parser.parse_args()


if __name__ == "__main__":
    main()
    print("Hello World!")
