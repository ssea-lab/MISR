import argparse
import os

import numpy as np
import networkx as nx
import random

from gensim.models import Word2Vec


class Graph ():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len (walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted (G.neighbors (cur))
            if len (cur_nbrs) > 0:
                if len (walk) == 1:
                    walk.append (cur_nbrs[alias_draw (alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw (alias_edges[(prev, cur)][0],
                                                alias_edges[(prev, cur)][1])]
                    walk.append (next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list (G.nodes ())
        print ('Walk iteration:')
        for walk_iter in range (num_walks):
            print (str (walk_iter + 1), '/', str (num_walks))
            random.shuffle (nodes)
            for node in nodes:
                walks.append (self.node2vec_walk (walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted (G.neighbors (dst)):
            if dst_nbr == src:
                unnormalized_probs.append (G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge (dst_nbr, src):
                unnormalized_probs.append (G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append (G[dst][dst_nbr]['weight'] / q)
        norm_const = sum (unnormalized_probs)
        normalized_probs = [float (u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup (normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes ():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted (G.neighbors (node))]
            norm_const = sum (unnormalized_probs)
            normalized_probs = [float (u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup (normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges ():
                alias_edges[edge] = self.get_alias_edge (edge[0], edge[1])
        else:
            for edge in G.edges ():
                alias_edges[edge] = self.get_alias_edge (edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge (edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    K = len (probs)
    q = np.zeros (K)
    J = np.zeros (K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate (probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append (kk)
        else:
            larger.append (kk)

    while len (smaller) > 0 and len (larger) > 0:
        small = smaller.pop ()
        large = larger.pop ()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append (large)
        else:
            larger.append (large)

    return J, q


def alias_draw(J, q):
    K = len (J)

    kk = int (np.floor (np.random.rand () * K))
    if np.random.rand () < q[kk]:
        return kk
    else:
        return J[kk]

"""
def parse_args():
    parser = argparse.ArgumentParser (description="Run node2vec.")

    parser.add_argument ('--input', nargs='?', default='graph/karate.edgelist',
                         help='Input graph path')

    parser.add_argument ('--output', nargs='?', default='emb/karate.emb',
                         help='Embeddings path')

    parser.add_argument ('--dimensions', type=int, default=128,
                         help='Number of dimensions. Default is 128.')

    parser.add_argument ('--walk-length', type=int, default=80,
                         help='Length of walk per source. Default is 80.')

    parser.add_argument ('--num-walks', type=int, default=10,
                         help='Number of walks per source. Default is 10.')

    parser.add_argument ('--window-size', type=int, default=10,
                         help='Context size for optimization. Default is 10.')

    parser.add_argument ('--iter', default=1, type=int,
                         help='Number of epochs in SGD')

    parser.add_argument ('--workers', type=int, default=8,
                         help='Number of parallel workers. Default is 8.')

    parser.add_argument ('--p', type=float, default=1,
                         help='Return hyperparameter. Default is 1.')

    parser.add_argument ('--q', type=float, default=1,
                         help='Inout hyperparameter. Default is 1.')

    parser.add_argument ('--weighted', dest='weighted', action='store_true',
                         help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument ('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults (weighted=False)

    parser.add_argument ('--directed', dest='directed', action='store_true',
                         help='Graph is (un)directed. Default is undirected.')
    parser.add_argument ('--undirected', dest='undirected', action='store_false')
    parser.set_defaults (directed=False)

    return parser.parse_args ()
"""


class args(object):
    def __init__(self,node2vec_path):
        """
        :param node2vec_path: 存放node2vec各种结果的路径
        """
        self.node2vec_path=node2vec_path # '../coldstart/node2vec'
        self.input=os.path.join(node2vec_path,'UI.txt')
        self.m_id_map_path= os.path.join(node2vec_path,'m_id_map.csv')#
        self.a_id_map_path = os.path.join(node2vec_path,'a_id_map.csv')  #
        self.output=os.path.join(node2vec_path,'result.txt')
        self.m_embedding=os.path.join(node2vec_path,'m_embedding.txt')
        self.a_embedding = os.path.join(node2vec_path,'a_embedding.txt')

        self.dimensions=25
        self.walk_length=80
        self.num_walks=10
        self.window_size=10
        self.iter=1
        self.workers=8
        self.p=1
        self.q=1
        self.weighted = False
        self.directed = False


def read_graph(a_args):
    if a_args.weighted:
        G = nx.read_edgelist (a_args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph ())
    else:
        G = nx.read_edgelist (a_args.input, nodetype=int, create_using=nx.DiGraph ())
        for edge in G.edges ():
            G[edge[0]][edge[1]]['weight'] = 1

    if not a_args.directed:
        G = G.to_undirected ()

    return G


def learn_embeddings(a_args,walks):
    walks = [list(map (str, walk)) for walk in walks]
    model = Word2Vec (walks, size=a_args.dimensions, window=a_args.window_size, min_count=0, sg=1, workers=a_args.workers,
                      iter=a_args.iter)
    model.wv.save_word2vec_format(a_args.output)
    return


def call_node2vec(a_args):
    nx_G = read_graph (a_args)
    G = Graph (nx_G, a_args.directed, a_args.p, a_args.q)
    G.preprocess_transition_probs ()
    walks = G.simulate_walks (a_args.num_walks, a_args.walk_length)
    learn_embeddings (a_args,walks)


if __name__ == "__main__":
    """
    args = parse_args ()
    call_node2vec (args)
    """