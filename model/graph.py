import numpy as np

def connect2adjacency(connects, num_joint):
    adjacency = np.zeros((num_joint, num_joint))
    for (i, j) in connects:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    return adjacency

def normalize_graph(adjacency):
    Dl = np.sum(adjacency, 0)
    num_joint = adjacency.shape[0]
    Dn = np.zeros((num_joint, num_joint))
    for i in range(num_joint):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, adjacency), Dn)
    return DAD

class Graph():
    ''' The Graph for models
    
    Args:
        strategy: Select one of following adjacency strategies
        - unilabel
        - distance
        - spatial
        - part
        max_dis_connect: max connection distance
    '''
    def __init__(self, num_joints=18, strategy='spatial', max_dis_connect=1, parts_num=4):
        self.strategy = strategy

        self.num_joint = num_joints
        self.max_dis_connect = max_dis_connect
        self.parts_num = parts_num

        self.get_edge()
        self.get_adjacency()

    def get_edge(self):

        if self.num_joint == 18:
            self.center = 1
            self_connect = [(i, i) for i in range(self.num_joint)]
            self.self_connect = self_connect
            if self.strategy == 'part':
                head = [(0, 1), (1, 2), (1, 5), (0, 14), (14, 16), (0, 15), (15, 17)]
                lefthand = [(5, 6), (6, 7)]
                righthand = [(2, 3), (3, 4)]
                torso = [(1, 2), (1, 5), (2, 8), (5, 11)]
                leftleg = [(11, 12), (12, 13)]
                rightleg = [(8, 9), (9, 10)]
                self.parts = [head, lefthand, righthand, torso, leftleg, rightleg]
            else:
                neighbor_connect = [(0, 1), (0,14), (0,15), (1,2), (1,5), (2,3), (2,8), (3,4), (5,6), (5,11), (6,7), (8,9), (9,10),
                         (11, 12), (12, 13), (14,16), (15,17)]
                self.edge = self_connect + neighbor_connect
        else:
            raise ValueError(f'[ERROR] graph with dimension {self.dimension} not exist.')

    def get_adjacency(self):
        if self.strategy == 'part':
            A = []
            A.append(connect2adjacency(self.self_connect, self.num_joint))
            for p in self.parts:
                A.append(normalize_graph(connect2adjacency(p, self.num_joint)))
            self.A = np.stack(A)
        else:
            adjacency = np.zeros((self.num_joint, self.num_joint))
            for i, j in self.edge:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
            dis_matrix = np.zeros((self.num_joint, self.num_joint)) + np.inf
            trans_matrix = [
                np.linalg.matrix_power(adjacency, p)
                for p in range(self.max_dis_connect + 1)
            ]
            N = np.zeros((self.num_joint, self.num_joint))
            for dis in range(self.max_dis_connect, -1, -1):
                dis_matrix[trans_matrix[dis] > 0] = dis
                N[trans_matrix[dis] > 0] = 1
            N = N / np.sum(N, 0)

            if self.strategy == 'unilabel':
                self.A = N[np.newaxis, :]
            elif self.strategy == 'distance':
                A = np.zeros(
                    (self.max_dis_connect + 1, self.num_joint, self.num_joint))
                for dis in range(self.max_dis_connect + 1):
                    A[dis][dis_matrix == dis] = N[dis_matrix == dis]
                self.A = A
            elif self.strategy == 'spatial':
                A = []
                for dis in range(self.max_dis_connect + 1):
                    root = np.zeros((self.num_joint, self.num_joint))
                    close = np.zeros((self.num_joint, self.num_joint))
                    further = np.zeros((self.num_joint, self.num_joint))
                    for i in range(self.num_joint):
                        for j in range(self.num_joint):
                            if dis_matrix[i, j] == dis:
                                if dis_matrix[i, self.center] == dis_matrix[
                                        j, self.center]:
                                    root[i, j] = N[i, j]
                                elif dis_matrix[i, self.center] < dis_matrix[
                                        j, self.center]:
                                    close[i, j] = N[i, j]
                                else:
                                    further[i, j] = N[i, j]
                    if dis == 0:
                        A.append(root)
                    else:
                        A.append(root + close)
                        A.append(further)
                self.A = np.stack(A)
            else:
                raise ValueError(f'[Error] Strategy {self.strategy} not existing.')