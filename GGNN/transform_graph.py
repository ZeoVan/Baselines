import os
import sys
import argparse

sys.path.append("./utils")

from bidict import bidict
from pathlib import Path

regex = '\\"\s+([^"]+)\s+\\"'
excluded_tokens = [",", "{", ";", "}", ")", "(", '"', "'", "`", "", " ", "[]", "[", "]", "/", ":", ".", " "]
parser = argparse.ArgumentParser()
parser.add_argument("--node_type_path",
                    default="./nodetype.txt", type=str, help="Type vocab")
parser.add_argument("--node_token_path",
                    default="./token.txt", type=str, help="Token vocab")
parser.add_argument(
    "--input", default="./im_test", type=str, help="Input path")
parser.add_argument(
    "--output", default="./im_test_graph", type=str, help="Output path")

args = parser.parse_args()

if not os.path.exists(args.output):
    Path(args.output).mkdir(parents=True, exist_ok=True)


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
    input_path = args.input
    output_path = args.output

    node_type_lookup = {}
    node_token_lookup = {}

    token_vocabulary_path = args.node_token_path
    node_type_vocabulary_path = args.node_type_path

    with open(token_vocabulary_path, "r") as f1:
        data = f1.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_token_lookup[splits[1]] = int(splits[0])

    with open(node_type_vocabulary_path, "r") as f2:
        data = f2.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_type_lookup[splits[1]] = int(splits[0])


    node_type_lookup = bidict(node_type_lookup)
    node_token_lookup = bidict(node_token_lookup)

    for subdir, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".txt"):
                graphs_path = os.path.join(subdir, file)

                single_graph_file = []

                #build node_id -> node_type,token dic
                graph_node_type_lookup = {}
                graph_node_token_lookup = {}
                with open(graphs_path, "r") as raw_graph:
                    data = raw_graph.readlines()
                    # 3514:IF,'if ' 1 3500:CONDITION,'('
                    for line in data:
                        line = line.replace("\n", "")
                        line = line.replace("'", "")
                        line = " ".join(line.split())
                        splits = line.split(" ")
                        if splits[0] != "?" and len(splits) == 3:
                            # source_splits: 3514:IF 'if '
                            source_splits = splits[0].split(",")
                            if len(source_splits)>1:# have token
                                source_node_tokens =source_splits[1]
                                if len(source_splits[0].split(":"))>1:# have type
                                    source_node_id = source_splits[0].split(":")[0]
                                    source_node_type = source_splits[0].split(":")[1]
                                    if not source_node_id in graph_node_type_lookup.keys():
                                        graph_node_type_lookup[source_node_id] = source_node_type
                                    if not source_node_id in graph_node_token_lookup.keys():
                                        graph_node_token_lookup[source_node_id] = source_node_tokens
                            else:
                                if len(source_splits[0].split(":"))>1:# have type
                                    source_node_id = source_splits[0].split(":")[0]
                                    source_node_type = source_splits[0].split(":")[1]
                                    if not source_node_id in graph_node_type_lookup.keys():
                                        graph_node_type_lookup[source_node_id] = source_node_type
                            sink_splits = splits[2].split(",")
                            if len(sink_splits) > 1:  # have token
                                sink_node_tokens = sink_splits[1]
                                if len(sink_splits[0].split(":")) > 1:  # have type
                                    sink_node_id = sink_splits[0].split(":")[0]
                                    sink_node_type = sink_splits[0].split(":")[1]
                                    if not sink_node_id in graph_node_type_lookup.keys():
                                        graph_node_type_lookup[sink_node_id] = sink_node_type
                                    if not sink_node_id in graph_node_token_lookup.keys():
                                        graph_node_token_lookup[sink_node_id] = sink_node_tokens
                            else:
                                if len(sink_splits[0].split(":")) > 1:  # have type
                                    sink_node_id = sink_splits[0].split(":")[0]
                                    sink_node_type = sink_splits[0].split(":")[1]
                                    if not sink_node_id in graph_node_type_lookup.keys():
                                        graph_node_type_lookup[sink_node_id] = sink_node_type


                with open(graphs_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:

                        line = line.replace("\n", "")
                        line = line.replace("'", "")
                        line = " ".join(line.split())


                        new_line_arr = []
                        splits = line.split(" ")
                        if splits[0] != "?":
                            if "dummy" not in line:
                                if len(splits) == 3:
                                    source = splits[0]
                                    source_splits = source.split(",")
                                    source_node_id = source_splits[0].split(":")[0]
                                    if len(source_splits[0].split(":"))>1:
                                        source_node_type = source_splits[0].split(":")[1]
                                        source_node_type_id = node_type_lookup[source_node_type]
                                    else:
                                        if source_node_id in graph_node_type_lookup.keys():
                                            source_node_type = graph_node_type_lookup[source_node_id]
                                        source_node_type_id = node_type_lookup[source_node_type]

                                    sink = splits[2]
                                    sink_splits = sink.split(",")
                                    sink_node_id = sink_splits[0].split(":")[0]
                                    if len(sink_splits[0].split(":"))>1:
                                        sink_node_type = sink_splits[0].split(":")[1]
                                        sink_node_type_id = node_type_lookup[sink_node_type]
                                    else:
                                        if sink_node_id in graph_node_type_lookup.keys():
                                            sink_node_type = graph_node_type_lookup[sink_node_id]
                                        sink_node_type_id = node_type_lookup[sink_node_type]

                                    source_subtokens_str = "-"
                                    sink_subtokens_str = "-"
                                    if len(source_splits) == 2:
                                        source_token = source_splits[1]
                                        source_token = process_token(source_token)
                                        # print("Source token : "  + source_token)
                                        source_subtokens = source_token.split()
                                        source_subtokens_ids = []
                                        for source_subtoken in source_subtokens:
                                            if source_subtoken and source_subtoken not in excluded_tokens:
                                                if source_subtoken in node_token_lookup:
                                                    source_subtoken_id = node_token_lookup[source_subtoken]
                                                else:
                                                    source_subtoken_id = node_token_lookup["<SPECIAL>"]
                                                source_subtokens_ids.append(str(source_subtoken_id))
                                        if len(source_subtokens_ids) > 0:
                                            source_subtokens_str = "-".join(source_subtokens_ids)
                                    elif source_node_id in graph_node_token_lookup.keys():
                                        source_token = graph_node_token_lookup[source_node_id]
                                        source_token = process_token(source_token)
                                        source_subtokens = source_token.split()
                                        source_subtokens_ids = []
                                        for source_subtoken in source_subtokens:
                                            if source_subtoken and source_subtoken not in excluded_tokens:
                                                if source_subtoken in node_token_lookup:
                                                    source_subtoken_id = node_token_lookup[source_subtoken]
                                                else:
                                                    source_subtoken_id = node_token_lookup["<SPECIAL>"]
                                                source_subtokens_ids.append(str(source_subtoken_id))
                                        if len(source_subtokens_ids) > 0:
                                            source_subtokens_str = "-".join(source_subtokens_ids)

                                    if len(sink_splits) == 2:
                                        sink_token = sink_splits[1]
                                        sink_token = process_token(sink_token)
                                        # print("Sink token : " + sink_token)
                                        sink_subtokens = sink_token.split()
                                        sink_subtokens_ids = []
                                        for sink_subtoken in sink_subtokens:
                                            if sink_subtoken and sink_subtoken not in excluded_tokens:
                                                if sink_subtoken in node_token_lookup:
                                                    sink_subtoken_id = node_token_lookup[sink_subtoken]
                                                else:
                                                    sink_subtoken_id = node_token_lookup["<SPECIAL>"]
                                                sink_subtokens_ids.append(str(sink_subtoken_id))

                                        if len(sink_subtokens_ids) > 0:
                                            sink_subtokens_str = "-".join(sink_subtokens_ids)
                                    elif sink_node_id in graph_node_token_lookup.keys():
                                        sink_token = graph_node_token_lookup[sink_node_id]
                                        sink_token = process_token(sink_token)
                                        # print("Sink token : " + sink_token)
                                        sink_subtokens = sink_token.split()
                                        sink_subtokens_ids = []
                                        for sink_subtoken in sink_subtokens:
                                            if sink_subtoken and sink_subtoken not in excluded_tokens:
                                                if sink_subtoken in node_token_lookup:
                                                    sink_subtoken_id = node_token_lookup[sink_subtoken]
                                                else:
                                                    sink_subtoken_id = node_token_lookup["<SPECIAL>"]
                                                sink_subtokens_ids.append(str(sink_subtoken_id))

                                        if len(sink_subtokens_ids) > 0:
                                            sink_subtokens_str = "-".join(sink_subtokens_ids)

                                    edge_id = splits[1]
                                    new_line_arr.append(str(source_node_id))
                                    new_line_arr.append(str(source_node_type_id))
                                    new_line_arr.append(str(source_subtokens_str))

                                    new_line_arr.append(edge_id)

                                    new_line_arr.append(str(sink_node_id))
                                    new_line_arr.append(str(sink_node_type_id))
                                    new_line_arr.append(str(sink_subtokens_str))

                                    single_graph_file.append(",".join(new_line_arr))
                        else:
                            # ? e/sample_data/java-small/training/project_2/Actor_getParent.java
                            #? e/test_cpp/0/118697.cpp
                            splits = line.split(" ")
                            file_path_splits = splits[1].split("/")
                            file_name = file_path_splits[len(file_path_splits) - 1]
                            lable_name = file_path_splits[len(file_path_splits) - 2]
                            single_graph_file.append("? " + str(lable_name))

                            file_name = file_name.replace(".cpp", ".txt")
                            file_name_with_lable = lable_name + "_" + file_name
                            path_with_lable = output_path+"/"+lable_name
                            new_file_path = os.path.join(path_with_lable, file_name_with_lable)
                            print(new_file_path)
                            try:
                                with open(new_file_path, "w") as f4:
                                    for new_line in single_graph_file:
                                        f4.write(new_line)
                                        f4.write("\n")
                            except Exception as e:
                                print(e)
                                new_file_path = os.path.join(output_path, file_name)
                                with open(new_file_path, "w") as f4:
                                    for new_line in single_graph_file:
                                        f4.write(new_line)
                                        f4.write("\n")

                            # Reset the graph file object
                            single_graph_file = []


if __name__ == "__main__":
    main()