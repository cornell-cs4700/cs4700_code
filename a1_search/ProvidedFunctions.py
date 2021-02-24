"""Functions provided to students. Their implementation should be hidden by
providing only compiled versions.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygraphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch.nn import CrossEntropyLoss

def compare_output_test(test_vals, result):
    pass

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

    def __str__(self):
        
        def str_(node, indent="", depth=10):
            """Returns a string representation of [node]."""
            if depth == 0:
                return f"{indent}{node.id} - {node.value}"
            else:
                child_strs = "".join([str_(c, indent=indent + "    ", depth=depth-1) for c in node.children])
                return f"{indent}{node.id} - {node.value}\n{child_strs}"

        return str_(self)
        
    def __eq__(self, other): return str(self) == str(other)


def to_networkx(node, graph=None, parent_id=None, depth=10):
    """...."""
    if depth > 0:
        graph = nx.Graph() if graph is None else graph
        graph.add_node(node.id)
        if parent_id is not None:
            graph.add_edge(parent_id, node.id)

        for c in node.children:
            to_networkx(c, graph=graph, parent_id=node.id, depth=depth-1)

        return graph

def show(graph):
    graph = to_networkx(graph)
    nx.drawing.nx_agraph.graphviz_layout(graph)
    plt.show()

graph_test_2 = Node(0, 0, [
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

show(graph_test_2)



graph_1 = Node(0, 0, [])
# Catches issues with cycles
value_3, id_3 = 4, 4
n2 = Node(2, 2, [])
graph_test_3 = Node(0, 0, [
    Node(2, 2, []),
    n2,
    Node(3, 3, [
        Node(4, 4, [])
    ]),
])
n2.children = [graph_test_3]

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
    random.seed(seed)
    return [np.random.rand(15) for _ in range(size)]
    
    
class InfinityNode(object):
    """A class implementing an infinite graph.
    
    Each instance of the class contains a [children] attribute which appears to
    be a list of the children of the InfinityNode node. *This attribute is set
    lazily* via the __getattribute__ method!
    
    Hopefully, this makes DFS completely untenable, and allows for a scale in
    which IDS can defeat BFS!
    """
    # nodes = []
    cls_count = 0
    specified_id = -1
    specified_value = -1
    max_nodes = -1

    def __init__(self, cls_id=None, cls_value=None, max_nodes=None, create_new=True):
        """
        The first three arguments must be specified on initialization.

        Args:
        cls_id      -- the ID matched with [value]
        cls_value   -- the value for nodes with ID [id]
        max_nodes   -- the maximum number of nodes in the class
        create_new  -- whether or not to reset class variables. This should be
                        True when a new InfinityNode is constructed by the
                        user, and False when a new InfinityNode is made by
                        the create_n() method.
        """
        if create_new:
            # self.__class__.nodes = [self]
            self.__class__.cls_count = 0
        else:
            pass
            # self.__class__.nodes.append(self)
        
        if cls_id is not None and cls_value is not None and max_nodes is not None:
            self.__class__.specified_id = cls_id
            self.__class__.specified_value = cls_value
            self.__class__.max_nodes = max_nodes

        self.id = self.__class__.cls_count      # unique ID of the node
        self.__class__.cls_count += 1
        
        self.children = "Congrats if you figured out this attribute isn't a list! It's actually set to a list of InfinityNode nodes lazily with the __getattribute__ method."
        self.children_set = False
        self.value = np.random.rand()

        if self.id == self.__class__.specified_id:
            self.value = self.__class__.specified_value
    

    def __hash__(self): return self.id

    def __repr__(self): return self.__str__()

    def __str__(self):
        
        def str_(node, indent="", depth=5):
            """Returns a string representation of [node]."""
            if depth == 1:
                return f"{indent}{node.id} - {node.value}"
            else:
                child_strs = "\n".join([str_(c, indent=indent + "    ", depth=depth-1) for c in node.children])
                return f"{indent}{node.id} - {node.value}\n{child_strs}"
        
        depth = 6
        return f"Top {depth} layers of tree:\n{str_(self, depth=depth)}"

    def __getattribute__(self, attribute):
        # The [children] attribute is lazily evaluated here. When it is
        # accessed the first time, change it from a string to a list of
        # new nodes.
        if attribute == "children" and not self.children_set:
            self.children_set = True
            self.children = list(InfinityNode.create_n(seed=int(self.id)))
            return self.children
        else:
            return object.__getattribute__(self, attribute)

    @classmethod
    def create_n(cls, n=2, p_repeat=0, seed=None):
        """Returns a generator over [n] new nodes, with a [p_repeat] probability
        of repetition from [nodes].

        Args:
        n           -- the number of nodes to generate
        p_repeat    -- the probability of sampling from [nodes] the pool of
                        existing nodes. Deprecated.
        seed        -- the seed for the random processes used in this method
        """
        np.random.seed(seed)
        return (InfinityNode(create_new=False) for _ in range(n))