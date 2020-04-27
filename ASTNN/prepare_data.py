import pandas as pd
import os
import re
import sys
from gensim.models.word2vec import Word2Vec
import pickle
from tree import ASTNode, SingleNode
import numpy as np


def get_sequences(node, sequence):
    current = SingleNode(node)
    sequence.append(current.get_token())
    for child in list(node.get_children()):
        get_sequences(child, sequence)
    if current.get_token().lower() == 'cursorkind.compound_stmt':
        sequence.append('End')

def get_blocks(node, block_seq):
    children = list(node.get_children())
    name = str(node.kind)
    if name in ['CursorKind.FUNCTION_DECL', 'CursorKind.IF_STMT', 'CursorKind.FOR_STMT', 'CursorKind.WHILE_STMT', 'CursorKind.DO_STMT']:
        block_seq.append(ASTNode(node))
        if name is not 'CursorKind.FOR_STMT':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i]
            if str(child.kind) not in ['CursorKind.FUNCTION_DECL', 'CursorKind.IF_STMT', 'CursorKind.FOR_STMT', 'CursorKind.WHILE_STMT', 'CursorKind.DO_STMT', 'CursorKind.COMPOUND_STMT']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
    elif name is 'CursorKind.COMPOUND_STMT':
        block_seq.append(ASTNode(name))
        for child in list(node.get_children()):
            if str(child.kind) not in ['CursorKind.IF_STMT', 'CursorKind.FOR_STMT', 'CursorKind.WHILE_STMT', 'CursorKind.DO_STMT']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
        block_seq.append(ASTNode('End'))
    else:
        for child in list(node.get_children()):
            get_blocks(child, block_seq)