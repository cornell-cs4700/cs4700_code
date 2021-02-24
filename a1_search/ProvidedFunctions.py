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

def str_(node, indent="", depth=10, root=False, last=False):
    """Returns a string representation of [node]."""
    if depth == 0:
        return f"{indent}├───[{node.id}: {node.value}]"
    elif root:
        child_strs = "".join([str_(c, indent=indent + " " * (1+len(str(node.id))), depth=depth-1, last=i+1==len(node.children))
                                            for i,c in enumerate(node.children)])
        return f"[{node.id}: {node.value}]\n{child_strs}"
    elif last:
        child_strs = "".join([str_(c, indent=indent + "│     ", depth=depth-1, last=i+1==len(node.children))
                                            for i,c in enumerate(node.children)])
        return f"{indent}└───[{node.id}: {node.value}]\n{child_strs}"
    else:
        child_strs = "".join([str_(c, indent=indent + "│     ", depth=depth-1, last=i+1==len(node.children))
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

    def __str__(self): return str_(self, root=True)
        
    def __eq__(self, other): return str_(self) == str(other)
    
graph_3 = Node(0, 0, [
    Node(1, 1, []),
    Node(2, 2, [
        Node(3, 3, []),
        Node(4, 4, [])
    ]),
    Node(5, 5, [])
])

print(graph_3)

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

def get_successor(state, n_successors=100):
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
    return [state - grad + (np.random.rand(15) * grad_norm / 2) for _ in range(n_successors)]

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