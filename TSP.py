import numpy as np
import random
import math


class City:
	def __init__(self):
		self.x = random.randint(0, 50)
		self.y = random.randint(0, 50)

	def __str__(self):
		return '({},{})'.format(self.x, self.y)


def get_city(n: int):
	city_list = [i for i in range(n)]
	return city_list


def get_dis(x: City, y: City):
	return math.sqrt(math.pow(x.x-y.x, 2) + math.pow(x.y-y.y, 2))


def all_dis(_list:list, city_list): # _list is a index list of the origin list
	dis = 0
	for i in range(len(_list)-1):
		dis += get_dis(city_list[_list[i]], city_list[_list[i+1]])
	dis += get_dis(city_list[_list[-1]], city_list[_list[0]])
	return dis


def get_permutation(array):
	_list = []
	for i in range(len(array)):
		_list.append(np.nanargmax(array[i]))
	return _list


def all_dis_binary(x, citys):
	return all_dis(get_permutation(x), citys)


def SA(citys):
	n = len(citys)  # number of cities
	gen_t = 100 * n  # number of generations
	#
	x = get_city(n)
	dis = 99999999
	dis_best = dis
	answer = ''
	T = 280
	rate = 0.92
	for i in range(gen_t):
		# random.shuffle(x)  # give it a initial state.  random method
		idx_1 = -1
		idx_2 = -1
		while idx_1 == idx_2:
			idx_1 = random.randint(0, len(x) - 1)
			idx_2 = random.randint(0, len(x) - 1)
			x[idx_1], x[idx_2] = x[idx_2], x[idx_1]

		new_dis = all_dis(x, citys)  # compute the new distance violently

		if dis > new_dis:
			dis = new_dis
			if dis_best > dis:
				dis_best = dis
				answer = x.__str__()
		elif dis == new_dis:
			break
		else:
			P = math.exp(-1 * (new_dis - dis) / T)
			if random.random() < P:
				dis = new_dis
		T *= rate
		if i % 100 == 0:
			print('distance:{}, SA answer:{}'.format(dis, answer))
	return dis


def GA_binary(citys):
	n = len(citys)
	population = 50
	gen_t = 100*n
	p = 0.01
	# citys = [City() for i in range(n)]
	peoples = []

	_list = [i for i in range(n)]

	min_dis = 99999999
	answer = ''
	for _ in range(population):
		code = np.array([[0]*n]*n)
		random.shuffle(_list)
		for i in range(len(code)):
			code[i][_list[i]] = 1  # initial
		peoples.append(code)

	for _ in range(gen_t):
		p_list = []
		min_list = []
		peoples = sorted(peoples, key=lambda x: all_dis_binary(x, citys))

		for i in peoples[0]:
			min_list.append(np.argmax(i))
		new_dis = all_dis(min_list, citys)
		if min_dis > new_dis:
			min_dis = new_dis
			answer = min_list.__str__()
		if _ % 100 == 0:
			print('distance:{}, GA answer:{}'.format(min_dis, answer))

		_sum = 0

		for i in peoples:
			_dis = all_dis_binary(i, citys)
			p_list.append(_dis)
			_sum += _dis
		for i in range(len(p_list)):
			p_list[i] /= _sum

		index_list = np.random.choice([i for i in range(population)], size=2, p=p_list)
		father = peoples[index_list[0]]
		mom = peoples[index_list[1]]
		son = []
		mating_index = random.randint(0, n-1)
		for i in father[:mating_index]:
			son.append(i)
		for i in range(0, len(mom)):
			flag = True
			for j in son:
				_code = np.argmax(j)
				if np.argmax(mom[i]) == _code:
					flag = False
					break
			if flag:
				son.append(mom[i])

		i_idx = -1  # genic mutation
		j_idx = -1
		if random.random() < p:
			while i_idx == j_idx:
				i_idx = random.randint(0, n-1)
				j_idx = random.randint(0, n-1)
			son[i_idx], son[j_idx] = son[j_idx], son[i_idx]  # swap

		son = np.array(son)

		peoples.pop(-1)
		peoples.append(son)

	return min_dis


if __name__ == '__main__':
	test_num = 20
	n = 10
	GA_better_than_SA = 0
	for _ in range(test_num):
		citys = [City() for i in range(n)]
		dis1 = GA_binary(citys)
		dis2 = SA(citys)
		if dis1 < dis2:
			GA_better_than_SA += 1
	print(GA_better_than_SA/test_num)
