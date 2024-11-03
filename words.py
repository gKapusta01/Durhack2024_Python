import itertools
import json
import os.path
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from nltk import FreqDist
from nltk.corpus import wordnet
from nltk.corpus import brown
import pprint
#import nltk
#nltk.download('words')
#nltk.download('brown')
#nltk.download('wordnet')

# -----------------------------------
# WORD PROCESSING

DATA_FOLDER = Path("data")
WORDS_ALL_PATH = DATA_FOLDER / "all_words.json"
WORDS_FREQUENCIES_PATH = DATA_FOLDER / "words_frequency.pkl"
WORDS_DEFINITIONS_PATH = DATA_FOLDER / "words_definitions.json"

def get_definitions(word: str):
    syns = wordnet.synsets(word)
    return [sys.definition() for sys in syns]

def create_words_dataset():
    # Get all possible words
    words_list = [word.lower() for word in brown.words() if word.isalpha()]

    # Remove words that don't have any definition
    words_definitions = dict()
    for word in words_list:
        if word in words_definitions:
            continue
        definitions = get_definitions(word)
        if definitions:
            words_definitions[word] = definitions
    words_list = [word for word in words_list if word in words_definitions]
    words_freq = FreqDist(words_list)
    words_list = sorted(set(words_list))

    # Save words set, definitions and frequency to file
    with open(WORDS_ALL_PATH, "w") as file:           # Words
        json.dump(words_list, file)
    with open(WORDS_FREQUENCIES_PATH, "wb") as file:     # Frequency Distribution
        pickle.dump(words_freq, file)
    with open(WORDS_DEFINITIONS_PATH, "w") as file:
        json.dump(words_definitions, file)
    print('dataset created successfully')

def load_words_dataset():
    if not os.path.exists(WORDS_ALL_PATH) or \
       not os.path.exists(WORDS_FREQUENCIES_PATH) or \
       not os.path.exists(WORDS_DEFINITIONS_PATH):
        create_words_dataset()

    with open(WORDS_ALL_PATH, "r") as f:
        words_all: list[str] = json.load(f)
    with open(WORDS_FREQUENCIES_PATH, "rb") as f:
        words_freq: FreqDist = pickle.load(f)
    with open(WORDS_DEFINITIONS_PATH, "r") as f:
        words_definitions: dict[str, list[str]] = json.load(f)

    return words_all, words_freq, words_definitions

# -----------------------------------
# GRAPH PROCESSING

GRAPH_FOLDER = Path("graph")
GRAPH_SINGLE_ADDITION = GRAPH_FOLDER / "single_addition_graph.graphml"

def plot_graph(G: nx.Graph):
    # Draw the directed graph
    plt.figure(figsize=(20, 20))
    nx.draw(G, with_labels=True, node_color="skyblue", font_weight="bold",
            node_size=2000, font_size=12, arrows=True, arrowstyle="-|>")
    plt.show()

def plot_random_subgraph(G: nx.Graph, n: int):
    sampled_nodes = random.sample(list(G.nodes), n)
    subG = G.subgraph(sampled_nodes).copy()
    plot_graph(subG)

def plot_reachable(G: nx.Graph, start: str):
    reachable_nodes = nx.descendants(G, start) | {start}
    subgraph = G.subgraph(reachable_nodes)
    plot_graph(subgraph)

def test_graph():
    G = nx.DiGraph()

    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    G.add_nodes_from(names)

    edges = [("Alice", "Bob"), ("Bob", "Charlie"), ("Alice", "Diana"),
             ("Diana", "Eve"), ("Eve", "Frank"), ("Charlie", "Frank")]
    G.add_edges_from(edges)

    plot_graph(G)

def remove_all_letters(word: str):
    return [word[:i] + word[i+1:] for i in range(len(word))]

def create_single_addition_graph():
    # Makes a graph representing the moves we can make in the game
    # Removes words that are irrelevant to the game

    words_all, words_freq, words_definitions = load_words_dataset()

    # Every word starts as a node
    G = nx.DiGraph()

    # Now we go through all pairs of words to make edges
    # There are too many nodes to do this directly e.g. itertools.combinations(all_words, 2)
    # First optimisation idea:
    # Group all words by their length, only compare to those one larger
    # Second optimisation idea:
    # Instead of checking every word length n against n+1,
    # remove every letter from all n+1 words and see if it's in the n set

    length_words = dict()
    for word in words_all:
        length = len(word)
        if length not in length_words:
            length_words[length] = set()
        length_words[length].add(word)

    for length in length_words:
        if length-1 not in length_words:
            continue
        for word in length_words[length]:
            for sub_word in remove_all_letters(word):
                # TODO: use proper nltk stemming to see if a word is a plural or not
                # if sub_word + "s" == word:
                #     continue
                if sub_word in length_words[length-1]:
                    G.add_node(sub_word)
                    G.add_node(word)
                    G.add_edge(sub_word, word)

    nx.write_graphml(G, GRAPH_SINGLE_ADDITION)

def load_single_addition_graph() -> nx.DiGraph:
    if not os.path.exists(GRAPH_SINGLE_ADDITION):
        create_single_addition_graph()

    return nx.read_graphml(GRAPH_SINGLE_ADDITION)

# -----------------------------------
# PATH PROCESSING

GRAPH_SINGLE_ADDITION_ALL_PATHS = GRAPH_FOLDER / "single_addition_all_paths.json"

def create_all_paths():

    G = load_single_addition_graph()
    maximal_paths = []

    def dfs(node, path):
        nonlocal maximal_paths

        path.append(node)

        successors = list(G.successors(node))

        if len(successors) == 0 and len(path) > 1:
            maximal_paths.append(path.copy())

        for successor in successors:
            # No need to check for if we have been to this node before
            # as paths are monotonic increasing in word length
            dfs(successor, path)

        path.pop()

    # We can start paths at any node that doesn't have any edges coming into it
    source_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    for starting_node in source_nodes:
        dfs(starting_node, [])

    # Now we have all the paths, store them by their length
    length_paths = dict()
    for path in maximal_paths:
        length = len(path)
        if length not in length_paths:
            length_paths[length] = []
        length_paths[length].append(path)

    with open(GRAPH_SINGLE_ADDITION_ALL_PATHS, "w") as file:  # Words
        json.dump(length_paths, file)

def load_all_paths() -> dict[str, list[str]]:
    if not os.path.exists(GRAPH_SINGLE_ADDITION_ALL_PATHS):
        create_all_paths()

    with open(GRAPH_SINGLE_ADDITION_ALL_PATHS, "r") as f:
        all_paths: dict[str, list[str]] = json.load(f)

    return all_paths

paths = load_all_paths()
def get_paths_of_length(n: int):
    pprint.pprint(paths[str(n)])

get_paths_of_length(8)
