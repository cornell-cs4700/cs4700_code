"""Functions provided to students. Their implementation should be hidden by
providing only compiled versions.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch.nn import CrossEntropyLoss

def compare_output_test(test_vals, result):
    pass

def print_node(node, indent="", depth=10, root=False, last=False):
    """Returns a string representation of [node]."""
    if depth == 0 and not last:
        return f"{indent}├───[{node.id}: {node.value}]\n"
    elif depth == 0 and last:
        return f"{indent}└───[{node.id}: {node.value}]\n"
    elif root:
        child_strs = "".join([print_node(c, indent=indent + " " * (1+len(str(node.id))), depth=depth-1, last=i+1==len(node.children))
                                            for i,c in enumerate(node.children)])
        return f"[{node.id}: {node.value}]\n{child_strs}"
    elif last:
        child_strs = "".join([print_node(c, indent=indent + "│     ", depth=depth-1, last=i+1==len(node.children))
                                            for i,c in enumerate(node.children)])
        return f"{indent}└───[{node.id}: {node.value}]\n{child_strs}"
    else:
        child_strs = "".join([print_node(c, indent=indent + "│     ", depth=depth-1, last=i+1==len(node.children))
                                            for i,c in enumerate(node.children)])
        return f"{indent}├───[{node.id}: {node.value}]\n{child_strs}"

class Node:
    """A basic implementation of a Node class that will hopefully help in
    debugging. Because it's not infinite, you have to construct the entire graph
    by hand, but hopefully hand-designed examples will help you tease out any
    errors in your solution!

    --- Example: ---------------------------------------------------------------
    tree = Node(0, 0, [
        Node(1, 1, [
            Node(3, 3, [Node(5, 5, []), Node(6, 6, [])]),
            Node(4, 4, [])
        ]),
        Node(2, 2, [Node(7, 7, []), Node(8, 8, [])]),
    ])

    has a BFS iteration order of 0, 1, 2, 3, 4, 7, 8, 5, 6 and a DFS iteration
    order of 0, 1, 3, 4, 6, 4, 2, 7, 8.
    """
    
    def __init__(self, id, value, children):
        """
        Args:
        id          -- the id of the Node
        value       -- the value of the node
        children    -- a list of the Node children of the Node, or an empty list
                        if the Node is a leaf
        """
        self.children = children
        self.id = id
        self.value = value if value is not None else np.random.rand() * 1000
    
    def __hash__(self): return self.id

    def __repr__(self): return self.__str__()

    def __str__(self): return print_node(self, root=True)
        
    def __eq__(self, other): return print_node(self, root=True) == print_node(other, root=True)
    
graph_1 = Node(0, 0, [
    Node(1, 1, []),
    Node(2, 2, [
        Node(3, 3, []),
        Node(4, 4, [])
    ]),
    Node(5, 5, [])
])
graph_1_value           = 4
graph_1_bfs_id          = 4
graph_1_ids_id          = 4
graph_1_bfs_sequence    = [0, 1, 2, 5, 3, 4]
graph_1_ids_sequence    = [0, 1, 2, 5, 0, 1, 2, 3, 4]

graph_2 = Node(0, 42, [])
graph_2_value           = 30
graph_2_bfs_id          = None
graph_2_ids_id          = None
graph_2_bfs_sequence    = [0]
graph_2_ids_sequence    = [0]

graph_3 = Node(0, 0, [
    Node(1, 1, [
        Node(8, 8, [
            Node(9, 9, []),
            Node(10, 10, []),
        ]),
        Node(12, 12, [
            Node(13, 13, [
                Node(14, 14, []),
                Node(15, 15, []),
            ])
        ]),
        Node(16, 16, [
            Node(17, 17, [
                Node(18, 18, [
                    Node(19, 19, [])
                ])
            ])
        ])
    ]),
    Node(2, 2, [
        Node(4, 4, []),
        Node(5, 5, [
            Node(6, 6, []),
            Node(7, 7, [])
        ])
    ]),
    Node(3, 3, []),
    Node(11, 11, []),
])
graph_3_value           = 19
graph_3_bfs_id          = 19
graph_3_ids_id          = 19
graph_3_bfs_sequence    = [0, 1, 2, 3, 11, 8, 12, 16, 4, 5, 9, 10, 13, 17, 6, 7, 14, 15, 18, 19]
graph_3_ids_sequence    = [
    0, 1, 2, 3, 11,
    0, 1, 8, 12, 16, 2, 4, 5, 3, 11,
    0, 1, 8, 9, 10, 12, 13, 16, 17, 2, 4, 5, 6, 7, 3, 11,
    0, 1, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 2, 4, 5, 6, 7, 3, 11,
    0, 1, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19
]

n2 = Node(2, 2, [])
graph_4 = Node(0, 0, [
    Node(1, 1, []),
    n2,
    Node(3, 3, [
        Node(4, 4, [])
    ]),
])
n2.children = [graph_4]
graph_4_value           = 4
graph_4_bfs_id          = 4
graph_4_ids_id          = 4
graph_4_bfs_sequence    = [0, 1, 2, 3, 0, 4]
graph_4_ids_sequence    = [0, 1, 2, 3, 0, 1, 2, 3, 4]

# We can markedly speed up the get_value() function by preloading data
x_train, _, y_train, _ = train_test_split(
        np.hstack([load_iris().data, np.ones((150,1))]),
        load_iris().target,
        random_state=1701,
        test_size=.5)

def get_value(state):
    """Returns the value of [state]"""
    w = np.reshape(state, (5,3))
    x,y = x_train, y_train
    fx = np.matmul(x, w)
    return np.sum(np.argmax(fx, axis=1) == y) / len(y)

def get_successor(state, n_successors=10):
    """Returns [n_successors] successor states to [state].
    
    The idea is to bias sampling in the direction opposite the gradient of the
    cross-entropy loss of [state].
    """
    loss_fn = CrossEntropyLoss()
    x,y = torch.tensor(x_train), torch.tensor(y_train)
    w = torch.tensor(np.reshape(state, (5,3)), requires_grad=True)
    fx = torch.matmul(x, w)
    loss = loss_fn(fx, y)
    loss.backward()
    
    grad = w.grad.numpy().reshape(15)
    grad_norm = np.linalg.norm(grad)
    successors = [state - grad + (np.random.rand(15) * grad_norm) for _ in range(n_successors)]
    successors += [state + grad + (np.random.rand(15) * grad_norm) for _ in range(n_successors)]
    return successors
    
def get_initial_state():
    """Returns an initial state."""
    return np.zeros(15)

def get_initial_population(size=100, seed=0):
    """Returns an initial population of [size] individuals, generated with seed
    [seed]. The real reason for doing this is that it forces students to
    properly seed their work, and GAs are sufficiently weird that it's good to
    optimize for a specific seed.
    """
    np.random.seed(seed)
    # random.seed(seed)
    return [np.random.rand(15) for _ in range(size)]