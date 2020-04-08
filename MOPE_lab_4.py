import numpy as np
from tabulate import tabulate
from scipy.stats import f, t, ttest_ind, norm
from _pydecimal import Decimal, ROUND_UP, ROUND_FLOOR
import copy


class Experiment():
    def __init__(self, m, N, q):
        self.m = m
        self.N = N
        self.q = q
        self.d = 0
        self.x = np.array([[-10, 50], [25, 65], [-10, 15]])
        self.x_min_avg = self.x[:, 0].mean()
        self.x_max_avg = self.x[:, 1].mean()
        self.y_max = 200 + self.x_max_avg
        self.y_min = 200 + self.x_min_avg

    def gen_mat(self):
        self.factors = []
        for i in range(2 ** self.x.shape[0]):
            self.factors.append(list(map(int, f'{i:0{self.x.shape[0]}b}')))
        rng = range(2 ** self.x.shape[0])
        to_pop = np.random.choice(
            2 ** self.x.shape[0], 2 ** self.x.shape[0] - self.N, replace=False)
        self.factors = np.delete(self.factors, to_pop, 0)
        self.nat_factors = np.copy(self.factors)
        tones = np.ones((self.factors.shape[0], 1), dtype=int)
        self.factors = np.hstack((tones, self.factors))
        self.factors = self.factors * 2 - 1

        self.nat_factors = self.nat_factors.T
        for i in range(self.x.shape[0]):
            arr = self.nat_factors[i]
            for j in range(len(arr)):
                if arr[j] == 1:
                    arr[j] = self.x[i][1]
                else:
                    arr[j] = self.x[i][0]
            self.nat_factors[i] = arr
        self.nat_factors = self.nat_factors.T

        self.y = np.random.sample((self.N, self.m)) * (self.y_max - self.y_min) + self.y_min

    def gen_matrix(self):
        self.factors = []
        for i in range(2 ** self.x.shape[0]):
            self.factors.append(list(map(int, f'{i:0{self.x.shape[0]}b}')))
        self.factors = np.array(self.factors)
        to_pop = np.random.choice(
            2 ** self.x.shape[0], 2 ** self.x.shape[0] - self.N, replace=False)
        self.factors = np.delete(self.factors, to_pop, 0)
        self.nat_factors = np.copy(self.factors).T
        tones = np.ones((self.factors.shape[0], 1), dtype=int)
        self.factors = np.hstack((tones, self.factors))
        self.factors[self.factors == 0] = -1

        for i, e in enumerate(self.nat_factors):
            for j, f in enumerate(e):
                e[j] = self.x[i][f]
                e[j] = self.x[i][f]
            self.nat_factors[i] = e
        self.nat_factors = self.nat_factors.T
        self.y = np.random.sample((self.N, self.m)) * \
                 (self.y_max - self.y_min) + self.y_min

    def def_params(self):
        self.y_mean = np.apply_along_axis(np.mean, 1, self.y)
        self.x_mean = np.apply_along_axis(np.mean, 0, self.nat_factors)
        self.y_mm = np.mean(self.y_mean)
        self.y_std = np.apply_along_axis(np.std, 1, self.y)

        a1, a2, a3 = 0, 0, 0
        a11, a22, a33 = 0, 0, 0
        self.a12, self.a13, self.a23 = 0, 0, 0
        j = 0
        for i in self.nat_factors:
            a1 += i[0] * self.y_mean[j]
            a2 += i[1] * self.y_mean[j]
            a3 += i[2] * self.y_mean[j]
            a11 += i[0] ** 2
            a22 += i[1] ** 2
            a33 += i[2] ** 2
            self.a12 += i[0] * i[1] / self.N
            self.a13 += i[0] * i[2] / self.N
            self.a23 += i[1] * i[2] / self.N
            j += 1
        self.a1, self.a2, self.a3 = a1 / self.N, a2 / self.N, a3 / self.N
        self.a11, self.a22, self.a33 = a11 / self.N, a22 / self.N, a33 / self.N
        self.a21 = self.a12
        self.a31 = self.a13
        self.a32 = self.a23

    def gen_b(self):
        start_matrix = np.array([
            [1, self.x_mean[0], self.x_mean[1], self.x_mean[2]],
            [self.x_mean[0], self.a11, self.a12, self.a13],
            [self.x_mean[1], self.a12, self.a22, self.a32],
            [self.x_mean[2], self.a13, self.a23, self.a33]
        ])
        start_det = np.linalg.det(start_matrix)

        to_replace = [self.y_mm, self.a1, self.a2, self.a3]
        self.b = []
        for i in range(4):
            new_matrix = np.copy(start_matrix)
            new_matrix = np.array(new_matrix)
            new_matrix[i] = to_replace
            self.b.append(np.linalg.det(new_matrix) / start_det)

    def gen_args(self):
        self.gen_mat()
        self.def_params()
        self.gen_b()

    def get_cohren_value(self):
        size_of_selections = self.N + 1
        qty_of_selections = self.m - 1
        significance = self.q
        partResult1 = significance / (size_of_selections - 1)
        params = [partResult1, qty_of_selections,
                  (size_of_selections - 1 - 1) * qty_of_selections]
        fisher = f.isf(*params)
        result = fisher / (fisher + (size_of_selections - 1 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    def get_student_value(self):
        f3 = (self.m - 1) * self.N
        significance = self.q
        result = abs(t.ppf(significance / 2, f3))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    def get_fisher_value(self):
        f3 = (self.m - 1) * self.N
        f4 = self.N - self.d
        significance = self.q
        result = abs(f.isf(significance, f4, f3))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    def cohren_crit(self):
        return (np.max(self.y_std) / np.sum(self.y_std)) < self.get_cohren_value()

    def student_crit_1(self):
        y_std_mean = np.mean(self.y_std)
        self.S_2b = y_std_mean / (self.N * self.m)
        t = []
        for i in range(4):
            t.append(
                np.abs(np.sum(self.y_mean * self.factors.T[i]) / self.N) / self.S_2b)
        self.dev = ""
        to_save = []
        t_t = self.get_student_value()
        for i in range(4):
            if t[i] > t_t:
                self.dev = self.dev + f"b{i} * x{i}"
                added = True
                to_save.append(1)
            else:
                print(f"Коефіцієнт b{i} - незначимий")
                added = False
                to_save.append(0)
            if i != 3 and added:
                self.dev = self.dev + " + "
        self.d = to_save.count(1)
        for i in range(4):
            if to_save[i] == 0:
                self.b[i] = 0
        self.to_compare = [self.b[0] + self.b[1] * i[0] + self.b[2]
                           * i[1] + self.b[3] * i[2] for i in self.nat_factors]

    def student_crit(self):
        y_std_mean = np.mean(self.y_std)
        self.S_2b = y_std_mean / (self.N * self.m)
        # b = np.mean(self.extended_real.T * self.y_mean, axis=1)
        b = [0] * self.N
        for i in range(self.N):
            for j in range(self.N - 1):
                b[i] += self.y_mean[i] * self.extended_real[i][j]
        t = np.array([np.abs(i) / np.sqrt(self.S_2b) for i in b])
        ret = np.where(t > self.get_student_value())
        for i in range(self.N):
            if i not in ret[0]:
                print(f"Коефіцієнт {self.b[i]} - незначимий")
                self.b[i] = 0
        self.d = len(ret[0])
        self.test = []
        for i in range(self.N):
            x = self.b[0]
            for j in range(1, 7):
                x += self.b[j] * self.extended_real[i][j]
            self.test.append(x)

    def fisher_crit(self):
        if self.d != self.N:
            S_2_ad = self.m / \
                     (self.N - self.d) * \
                     np.sum((np.array(self.y_mean) - np.array(self.to_compare)) ** 2)
            F_p = S_2_ad / self.S_2b
            return F_p < self.get_fisher_value()
        return True

    ###### Lab4 ######

    def define_new_params(self):
        self.gen_mat()
        self.y_mean = np.apply_along_axis(np.mean, 1, self.y)
        self.y_std = np.apply_along_axis(np.std, 1, self.y)
        self.seq = [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
        self.extended = []
        self.extended_real = []
        self.extended.append([1] * self.N)
        for i in self.seq:
            column = [1] * self.N
            column_real = [1] * self.N
            for j in i:
                column *= self.factors.T[j]
                column_real *= self.nat_factors.T[j - 1]
            self.extended.append(column)
            self.extended_real.append(column_real)
        self.extended_real = np.array(self.extended_real).T
        self.extended = np.array(self.extended).T

    def generate_new_matrix(self):
        seq1 = [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
        self.new_matrix = [[copy.copy(seq1[i]) for j in range(self.N)] for i in range(self.N)]
        self.new_matrix = np.array(self.new_matrix).T
        for i in range(self.N):
            for j in range(self.N):
                self.new_matrix[i][j].extend(seq1[i])
        self.new_matrix = self.new_matrix.T

    def find_for_norm(self):
        self.b = [0] * self.N
        for i in range(self.N):
            for j in range(self.N):
                self.b[j] += self.y_mean[i] * self.extended[i][j] / self.N
        return self.b

    def find_for_real(self):
        matrix = np.copy(self.new_matrix)
        self.b = [0] * self.N
        f = np.copy(self.new_matrix)
        ext_real = np.copy(self.extended_real)
        ext_real = np.insert(ext_real, 0, [1] * self.N, axis=1)

        for i in range(self.N):
            for j in range(self.N):
                s = [1] * self.N
                if f[i][j]:
                    for column in f[i][j]:
                        s *= self.nat_factors.T[column - 1]
                    f[i][j] = np.sum(s)
        f[0][0] = self.N
        f = f.astype('float64')
        det = np.linalg.det(f)
        k = [0 for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                k[j] += self.y_mean[i] * ext_real[i][j]
        self.k = k

        for i in range(self.N):
            m1 = np.copy(f).T
            m1[i] = k
            self.b[i] = np.linalg.det(m1) / det
        return self.b

    def print_data(self):
        col = []
        for i in range(len(self.y_mean)):
            col.append([self.y_mean[i], self.y_std[i]])
        to_print = np.hstack((self.extended_real, self.y, col))
        headers = ['x1', 'x2', 'x3', 'x1 * x2', 'x1 * x3', 'x2 * x3', 'x1 * x2 * x3']
        for i in range(self.m):
            headers.append(f'y{i + 1}')
        headers.append('y')
        headers.append('S(y)')
        print(tabulate(list(to_print), headers=headers, tablefmt="fancy_grid"))

        b_norm = self.find_for_norm()
        b_norm = np.round(b_norm, decimals=2)
        b_norms = np.sign(b_norm)
        b_norms = np.array(list(map(lambda x: "-" if x < 0 else "+", b_norms)))
        b_norma = np.abs(b_norm)
        b_real = self.find_for_real()
        b_real = np.round(b_real, decimals=2)
        b_reals = np.sign(b_real)
        b_reals = np.array(list(map(lambda x: "-" if x < 0 else "+", b_reals)))
        b_reala = np.abs(b_real)

        print("Рівняння регресії", f"y = {b_reals[0] if b_reals[0] == '-' else ''}{b_reala[0]} "
                                   f"{b_reals[1]} {b_reala[1]} * x1 "
                                   f"{b_reals[1]} {b_reala[2]} * x2 "
                                   f"{b_reals[1]} {b_reala[3]} * x3 "
                                   f"{b_reals[1]} {b_reala[4]} * x1 * x2 "
                                   f"{b_reals[1]} {b_reala[5]} * x1 * x3 "
                                   f"{b_reals[1]} {b_reala[6]} * x2 * x3 "
                                   f"{b_reals[1]} {b_reala[7]} * x1 * x2 * x3")
        print("Рівняння регресії для кодованих значень", f"y = {b_norms[0] if b_norms[0] == '-' else ''}{b_norma[0]} "
                                                         f"{b_norms[1]} {b_norma[1]} * x1 "
                                                         f"{b_norms[1]} {b_norma[2]} * x2 "
                                                         f"{b_norms[1]} {b_norma[3]} * x3 "
                                                         f"{b_norms[1]} {b_norma[4]} * x1 * x2 "
                                                         f"{b_norms[1]} {b_norma[5]} * x1 * x3 "
                                                         f"{b_norms[1]} {b_norma[6]} * x2 * x3 "
                                                         f"{b_norms[1]} {b_norma[7]} * x1 * x2 * x3")

    def model_1(self):
        self.define_new_params()
        self.generate_new_matrix()
        while True:
            if not self.cohren_crit():
                print("Дисперсія неоднорідна за критерієм Кохрена")
                self.define_new_params()
                self.generate_new_matrix()
            else:
                break

        print("Дисперсія однорідна за критерієм Кохрена")
        self.find_for_norm()
        self.find_for_real()

        self.student_crit()
        if self.fisher_crit():
            print("Рівняння регресії адекватно оригіналу")
        else:
            print("Рівняння регресії неадекватно оригіналу")
            self.model_1()  # Повторення ефекту взаємодії (пункт 2 у блок-схемі)

    def model(self):
        self.gen_args()
        while not self.cohren_crit():
            print(f"Дисперсія неоднорідна за критерієм Кохрена при m = {self.m}, збільшимо m")
            self.m += 1
            self.gen_args()

        print("Дисперсія однорідна")
        self.student_crit_1()
        print(f"Рівняння регресії - {self.dev}")
        if self.fisher_crit() == True:
            print("Рівняння регресії адекватно оригіналу")
        else:
            print("Рівняння регресії неадекватно оригіналу")
            self.model_1()  # Перехід до ефекту взаємодії (пункт 2 у блок-схемі)

        self.print_data()


m = Experiment(3, 8, 0.05)
m.model()

