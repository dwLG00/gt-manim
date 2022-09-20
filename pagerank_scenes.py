from manim import *
from math import sin, cos
import numpy as np
from reducible_markov import *
import random

nodes = [1, 2, 3, 4]
edges = [(1, 2), (1, 3), (2, 3), (2, 4)]
labels = {
	1: Tex("0"),
	2: Tex("1"),
	3: Tex("2"),
	4: Tex("3")
}

GRAPH = Graph(nodes, edges, labels=labels, layout="circular")
_adjacency_matrix = np.matrix([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
_markov_matrix = np.matrix([[0, 1/3, 0.5, 0], [0.5, 0, 0.5, 1], [0.5, 1/3, 0, 0], [0, 1/3, 0, 0]])

ADJACENCY_MATRIX = Matrix([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
MARKOV_MATRIX = Matrix([[0, 0.33, 0.5, 0], [0.5, 0, 0.5, 1], [0.5, 0.33, 0, 0], [0, 0.33, 0, 0]])
DEGREE_MATRIX = Matrix([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])
markov_chain = MarkovChain(4, [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0), (1, 3), (3, 1)], _markov_matrix)
MARKOV_GRAPH = MarkovChainGraph(markov_chain)

class MarkovTest(Scene):
	def construct(self):
		markov_copy = MARKOV_GRAPH.copy()
		self.play(Create(markov_copy))
		self.wait(2)

class GraphToMarkov(Scene):
	def construct(self):
		markov = MARKOV_GRAPH.copy()
		graph = GRAPH.copy()
		graph[1].set_color(RED)
		graph[2].set_color(RED)
		graph[3].set_color(RED)
		graph[4].set_color(RED)
		labels[1].set_color(WHITE)
		labels[2].set_color(WHITE)
		labels[3].set_color(WHITE)
		labels[4].set_color(WHITE)
		graph.next_to(markov, LEFT*0.5)

		self.play(Create(graph))
		self.wait(1)
		graph.target = markov
		self.play(FadeTransform(graph, markov))
		self.wait(2)

class InitialGraphTravel(Scene):
	def construct(self):
		'''
		WHITE = "#ece6e2"
		BLACK = "#343434"
		self.camera.background_color = WHITE

		Mobject.set_default(color=BLACK)
		Text.set_default(color=BLACK)
		Square.set_default(color=BLACK)
		Circle.set_default(color=BLACK)
		'''

		graph = GRAPH.copy()
		agent = Circle(radius=0.1, color=BLUE_B, fill_opacity=1)
		agent.move_to(graph[1])
		self.play(Create(graph))
		self.wait(1)
		self.play(FadeIn(agent))
		self.wait(2)

		self.play(agent.animate.move_to(graph[2]))
		self.wait(1)
		self.play(agent.animate.move_to(graph[3]))
		self.wait(1)

		g = Group(graph, agent)
		matrix = ADJACENCY_MATRIX.copy()
		self.play(g.animate.shift(LEFT*3))
		matrix.next_to(g, RIGHT)
		self.play(FadeIn(matrix))
		self.wait(1)

		markov_matrix = MARKOV_MATRIX.copy()
		markov_matrix.next_to(g, RIGHT)
		self.play(FadeOut(matrix), FadeIn(markov_matrix))
		self.wait(1)

		v1 = Matrix([["p_0"], ["p_1"], ["p_2"], ["p_3"]], left_bracket="(", right_bracket=")")
		v1.next_to(markov_matrix, RIGHT)
		labels = [MathTex(r"p_0"), MathTex(r"p_1"), MathTex(r"p_2"), MathTex(r"p_3")]
		for label in labels:
			label.set_color(BLACK)
		labels[0].move_to(graph[1])
		labels[1].move_to(graph[2])
		labels[2].move_to(graph[3])
		labels[3].move_to(graph[4])
		label_group = Group(*labels)
		self.play(FadeIn(v1), FadeIn(label_group), FadeOut(agent))
		self.wait(1)
		self.play(FadeOut(graph), FadeOut(label_group), FadeOut(v1), markov_matrix.animate.shift(LEFT*6))
		self.wait(1)
		self.play(markov_matrix.animate.scale(0.7))
		self.wait(0.5)

		equals = Tex('=')
		equals.next_to(markov_matrix, RIGHT)
		matrix.scale(0.7)
		matrix.next_to(equals, RIGHT)
		divide = MathTex(r'\div')
		divide.next_to(matrix, RIGHT)
		degree = DEGREE_MATRIX.copy()
		degree.scale(0.7)
		degree.next_to(divide, RIGHT)
		equation_group = Group(equals, matrix, divide, degree)

		matrix_labels = [Tex("Transition Matrix"), Tex("Adjacency Matrix"), Tex("Degree Matrix")]
		matrix_labels[0].next_to(markov_matrix, UP)
		matrix_labels[1].next_to(matrix, UP)
		matrix_labels[2].next_to(degree, UP)

		matrix_label_group = Group(*matrix_labels)

		self.play(FadeIn(equation_group), FadeIn(matrix_label_group))
		self.wait(2)

class MarkovBulk(Scene):
	def construct(self):
		def small_deviation(v):
			x, y = 0.1 - (random.random() / 5), 0.1 - (random.random() / 5)
			return v + np.array([x, y, 0])

		total_agents = 100
		iterations = 5

		graph = GRAPH.copy()
		agents = [Circle(radius=0.03, color=BLUE_B, fill_opacity=1) for _ in range(total_agents)]
		agent_list = [[], [], [], []]
		possible_moves = [[1, 2], [0, 2, 3], [0, 1], [1]]
		for agent in agents:
			container = random.randint(0, 3)
			agent_list[container].append(agent)
		for i in range(4):
			container = agent_list[i]
			node_point = graph[i+1].get_center()
			for agent in container:
				agent.move_to(small_deviation(node_point))

		labels = [MathTex("p = ", str(len(agent_list[i]) / total_agents)) for i in range(4)]
		labels[0].next_to(graph[1], RIGHT)
		labels[1].next_to(graph[2], UP)
		labels[2].next_to(graph[3], LEFT)
		labels[3].next_to(graph[4], DOWN)

		agents_group = Group(*agents)
		labels_group = Group(*labels)
		self.play(Create(graph), FadeIn(agents_group), FadeIn(labels_group))
		self.wait(2)

		for _ in range(iterations):
			animations = []
			new_agent_list = [[], [], [], []]
			for i in range(4):
				container = agent_list[i]
				for agent in container:
					dest = random.choice(possible_moves[i])
					new_agent_list[dest].append(agent)
					animations.append(agent.animate.move_to(small_deviation(graph[dest+1].get_center())))
			agent_list = new_agent_list
			new_labels = [MathTex("p = ", str(len(agent_list[i]) / total_agents)) for i in range(4)]
			new_labels[0].move_to(labels[0])
			new_labels[1].move_to(labels[1])
			new_labels[2].move_to(labels[2])
			new_labels[3].move_to(labels[3])
			new_labels_group = Group(*new_labels)
			self.play(*animations, FadeOut(labels_group), FadeIn(new_labels_group))
			labels = new_labels
			labels_group = new_labels_group
			self.wait(1)
		self.wait(2)
		graph_related = Group(agents_group, labels_group, graph)
		self.play(graph_related.animate.shift(LEFT))
		self.wait(1)

		arrow = MathTex(r'\Rightarrow')
		arrow.next_to(labels[0], RIGHT)
		v1_raw = np.array([[len(agent_list[0])/total_agents], [len(agent_list[1])/total_agents], [len(agent_list[2])/total_agents], [len(agent_list[3])/total_agents]])
		v1 = Matrix(v1_raw, left_bracket="(", right_bracket=")")
		v1.next_to(arrow, RIGHT)
		self.play(FadeIn(arrow), FadeIn(v1))
		self.wait(1)

		markov_matrix = MARKOV_MATRIX.copy()
		markov_matrix.move_to(graph_related)
		self.play(FadeOut(graph_related), FadeIn(markov_matrix), FadeOut(arrow), v1.animate.next_to(markov_matrix, RIGHT))
		self.wait(1)

		markov_product_group = Group(markov_matrix, v1)
		self.play(markov_product_group.animate.shift(LEFT))
		self.wait(1)

		equals = Tex('=')
		equals.next_to(markov_product_group, RIGHT)
		v2_raw = _markov_matrix * v1_raw
		v2_raw = np.round(v2_raw, decimals=2)
		v2 = Matrix(v2_raw, left_bracket="(", right_bracket=")")
		v2.next_to(equals, RIGHT)
		v1_copy = v1.copy()
		v1_copy.target = v2
		self.play(FadeIn(equals), MoveToTarget(v1_copy))
		self.wait(1)
		v1_copy.set_opacity(0)

		while True:
			v1_copy.set_opacity(0)
			self.play(FadeOut(v1), v2.animate.move_to(v1))
			self.wait(0.2)
			v1 = v2
			v1_raw = v2_raw

			v2_raw = _markov_matrix * v1_raw
			v2_raw = np.round(v2_raw, decimals=2)
			v2 = Matrix(v2_raw, left_bracket="(", right_bracket=")")
			v2.next_to(equals, RIGHT)
			v1_copy = v1.copy()
			v1_copy.target = v2
			self.play(MoveToTarget(v1_copy))
			self.wait(0.2)
			if abs(v1_raw[0] - v2_raw[0]) <= 0.03 and abs(v1_raw[1] - v2_raw[1]) <= 0.03 and abs(v1_raw[2] - v2_raw[2]) <= 0.03 and abs(v1_raw[3] - v2_raw[3]) <= 0.03:
				break
		self.wait(1)

		lamda = MathTex('\lambda')
		lamda.next_to(equals, RIGHT)
		v1_copy2 = v1.copy()
		v1_copy2.next_to(lamda, RIGHT)
		about_equals = MathTex(r'\approx')
		about_equals.next_to(v1, RIGHT)
		lamda_group = Group(about_equals, lamda, v1_copy2)
		self.play(FadeOut(equals), FadeOut(v2), FadeOut(v1_copy), FadeIn(lamda_group))
		self.wait(2)
