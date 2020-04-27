import os
from concurrent.futures import ProcessPoolExecutor
import copy
import sys
import argparse
import re

sys.path.append("../utils")
from tqdm import *

regex = '\\"\s+([^"]+)\s+\\"'
excluded_tokens = [",", "{", ";", "}", ")", "(", '"', "'", "`", "", " ", "[]", "[", "]", "/", ":", ".", " "]
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./all_data_raw_code", type=str, help="Input path")
parser.add_argument("--output", default="./token.txt", type=str, help="Output path")

args = parser.parse_args()


def exclude_tokens(all_vocabularies):
    temp_vocabs = []
    for vocab in all_vocabularies:
        if vocab not in excluded_tokens:
            temp_vocabs.append(vocab)
    return temp_vocabs


def process_token(token):
    for t in excluded_tokens:
        token = token.replace(t, "")
    return token


def main():
    # input = "../sample_data/java-small-graph/training"
    # output = "../preprocessed_data/token_vocabulary.csv"
    input_path = args.input
    output_path = args.output

    all_graph_paths = []
    for subdir, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".txt"):
                graphs_path = os.path.join(subdir, file)
                all_graph_paths.append(graphs_path)

    all_vocabularies = set()
    all_vocabularies.add("<ZERO>")
    print("Total number of paths:", len(all_graph_paths))
    for path in tqdm(all_graph_paths):
        print("Compute token for : " + str(path))
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # print(line)

                line = line.replace("\n", "")
                line = line.replace("'", "")
                line = " ".join(line.split())
                # line = strip(line)
                # line
                splits = line.split(" ")

                if "?" not in line:
                    if len(splits) == 3:
                        source = splits[0]
                        source_splits = source.split(",")

                        sink = splits[2]
                        sink_splits = sink.split(",")

                        if len(source_splits) == 2:
                            source_tokens = source_splits[1].split()
                            for source_token in source_tokens:
                                if source_token not in excluded_tokens:
                                    source_token = process_token(source_token)
                                    all_vocabularies.add(source_token)

                        if len(sink_splits) == 2:
                            sink_tokens = sink_splits[1].split()
                            for sink_token in sink_tokens:
                                if sink_token not in excluded_tokens:
                                    sink_token = process_token(sink_token)
                                    all_vocabularies.add(sink_token)
        print("Num vocabs : ", len(all_vocabularies))
    # all_vocabularies = exclude_tokens(all_vocabularies)
    all_vocabularies = list(all_vocabularies)
    # unique_vocabularies = []
    # for vocab in all_vocabularies:
    # 	if vocab not in unique_vocabularies:
    # 		unique_vocabularies.append(vocab)
    all_vocabularies.append("<SPECIAL>")

    with open(output_path, "w") as f1:
        for i, v in enumerate(all_vocabularies):
            f1.write(str(i) + "," + v)
            f1.write("\n")


if __name__ == "__main__":
    main()