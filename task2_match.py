import time
import random
import numpy as np

zero_threshold = 0.00000001

class KMNode(object):
    def __init__(self, id, exception = 0, match = None, visit = False):
        self.id = id
        self.exception = exception
        self.match = match
        self.visit = visit


class alog_KM(object):
    def __init__(self):
        self.matrix = None
        self.x_nodes = []
        self.y_nodes = []
        self.x_length = 0
        self.y_length = 0
        self.index_x = 0
        self.index_y = 1
        self.minz = float('inf')


    def __del__(self):
        pass

    def set_matrix(self, x_y_values):
        xs = set()
        ys = set()
        for x, y, value in x_y_values:
            xs.add(x)
            ys.add(y)

        if len(xs) < len(ys):
            self.index_x = 0
            self.index_y = 1
        else:
            self.index_x = 1
            self.index_y = 0
            xs, ys = ys, xs

        x_dic = {x: i for i, x in enumerate(xs)}
        y_dic = {y: j for j, y in enumerate(ys)}
        self.x_nodes = [KMNode(x) for x in xs]
        self.y_nodes = [KMNode(y) for y in ys]
        self.x_length = len(xs)
        self.y_length = len(ys)

        self.matrix = np.zeros((self.x_length, self.y_length))
        for row in x_y_values:
            x = row[self.index_x]
            y = row[self.index_y]
            value = row[2]
            x_index = x_dic[x]
            y_index = y_dic[y]
            self.matrix[x_index, y_index] = value

        for i in range(self.x_length):
            self.x_nodes[i].exception = max(self.matrix[i, :])


    def KM(self):
        for i in range(self.x_length):
            while True:
                self.minz = float('inf')
                self.set_false(self.x_nodes)
                self.set_false(self.y_nodes)

                if self.dfs(i):
                    break

                self.change_exception(self.x_nodes, -self.minz)
                self.change_exception(self.y_nodes, self.minz)


    def dfs(self, i):
        match_list = []
        while True:
            x_node = self.x_nodes[i]
            x_node.visit = True
            for j in range(self.y_length):
                y_node = self.y_nodes[j]
                if not y_node.visit:
                    t = x_node.exception + y_node.exception - self.matrix[i][j]
                    if abs(t) < zero_threshold:
                        y_node.visit = True
                        match_list.append((i, j))
                        if y_node.match is None:
                            self.set_match_list(match_list)
                            return True
                        else:
                            i = y_node.match
                            break
                    else:
                        if t >= zero_threshold:
                            self.minz = min(self.minz, t)
            else:
                return False
    
    def set_match_list(self, match_list):
        for i, j in match_list:
            x_node = self.x_nodes[i]
            y_node = self.y_nodes[j]
            x_node.match = j
            y_node.match = i

    def set_false(self, nodes):
        for node in nodes:
            node.visit = False

    def change_exception(self, nodes, change):
        for node in nodes:
            if node.visit:
                node.exception += change

    def get_connect_result(self):
        ret = []
        for i in range(self.x_length):
            x_node = self.x_nodes[i]
            j = x_node.match
            y_node = self.y_nodes[j]
            x_id = x_node.id
            y_id = y_node.id
            value = self.matrix[i][j]

            if self.index_x == 1 and self.index_y == 0:
                x_id, y_id = y_id, x_id
            ret.append((x_id, y_id, value))

        return ret

    def get_max_value_result(self):
        ret = 0
        for i in xrange(self.x_length):
            j = self.x_nodes[i].match
            ret += self.matrix[i][j]

        return ret


def KM_running(x_y_values):
    process = alog_KM()
    process.set_matrix(x_y_values)
    process.KM()
    return process.get_connect_result()


def test():
    values = []
    random.seed(0)
    for i in range(500):
        for j in range(1000):
            value = random.random()
            values.append((i, j, value))

    return KM_running(values)

if __name__ == '__main__':
    s_time = time.time()
    ret = test()
    values=[]
    for i in range(30):
        for j in range(3):
            values.append((i,j,np.random.randn()))
    print(values)
    print('answer')
    print(KM_running(values))