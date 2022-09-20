from manim import *

class CentralityMain(Scene):
	def construct(self):
		nodes = [1, 2, 3, 4]
		edges = [(1, 2), (2, 3), (3, 4), (1, 4)]
		g = Graph(nodes, edges, layout="circular")
		g.shift(LEFT*2)

		labels = [MathTex("v_0"), MathTex("v_1"), MathTex("v_2"), MathTex("v_3")]
		labels[0].next_to(g[1], RIGHT)
		labels[1].next_to(g[2], UP)
		labels[2].next_to(g[3], LEFT)
		labels[3].next_to(g[4], DOWN)
		labels_group = Group(*labels)

		#color the labels
		COLORS = [RED, GREEN, BLUE, YELLOW]
		for i in range(4):
			labels[i].set_color(COLORS[i])

		adj = Matrix([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
		vec = Matrix([["v_0"], ["v_1"], ["v_2"], ["v_3"]], left_bracket="(", right_bracket=")")
		vec.next_to(labels_group, RIGHT*2)
		vec.set_row_colors(*COLORS)

		#initialize
		self.play(Create(g), FadeIn(labels_group))
		self.wait(1)
		self.play(FadeIn(vec))
		self.wait(2)

		#Matrix Product
		new_labels = [MathTex("v_1",  "+", "v_3"), MathTex("v_0", "+", "v_2"), MathTex("v_1", "+", "v_3"), MathTex("v_0", "+", "v_2")]
		new_labels[0][0].set_color(COLORS[1])
		new_labels[0][2].set_color(COLORS[3])
		new_labels[1][0].set_color(COLORS[0])
		new_labels[1][2].set_color(COLORS[2])
		new_labels[2][0].set_color(COLORS[1])
		new_labels[2][2].set_color(COLORS[3])
		new_labels[3][0].set_color(COLORS[0])
		new_labels[3][2].set_color(COLORS[2])


		'''
		label_copies = []
		for i in range(4):
			new_labels[i].move_to(labels[i])
			label_copies.append([labels[i].copy(), labels[i].copy()])
			labels[i].target = new_labels[i]
		'''

		self.play(labels[0].animate.shift(RIGHT*0.5), labels[2].animate.shift(LEFT*0.5), vec.animate.shift(RIGHT))
		self.wait(1)
		new_labels[0].move_to(labels[0])
		new_labels[2].move_to(labels[2])
		new_labels[1].move_to(labels[1])
		new_labels[3].move_to(labels[3])

		for i in range(4):
			labels[i].target = new_labels[i]

		'''
		self.play(
			*[FadeOut(labels[i]) for i in range(4)],
			*[FadeIn(new_labels[i]) for i in range(4)]
		)
		'''
		self.play(*[MoveToTarget(labels[i]) for i in range(4)])
		self.wait(1)

		self.play(FadeOut(labels_group))
		self.wait(1)

		ng = Group(g, *new_labels)
		self.play(ng.animate.shift(LEFT).scale(0.7))
		self.wait(1)
		new_labels_group = Group(*new_labels)
		adj.next_to(new_labels_group, RIGHT)
		self.play(vec.animate.next_to(adj, RIGHT))
		self.wait(1)
		self.play(FadeIn(adj))
		self.wait(1)

		equations = [
			MathTex("v_0", "=", "v_1", "+", "v_3"),
			MathTex("v_1", "=", "v_0", "+", "v_2"),
			MathTex("v_2", "=", "v_1", "+", "v_3"),
			MathTex("v_3", "=", "v_0", "+", "v_2"),
		]

		equations[0][0].set_color(COLORS[0])
		equations[0][2].set_color(COLORS[1])
		equations[0][4].set_color(COLORS[3])
		equations[1][0].set_color(COLORS[1])
		equations[1][2].set_color(COLORS[0])
		equations[1][4].set_color(COLORS[2])
		equations[2][0].set_color(COLORS[2])
		equations[2][2].set_color(COLORS[1])
		equations[2][4].set_color(COLORS[3])
		equations[3][0].set_color(COLORS[3])
		equations[3][2].set_color(COLORS[0])
		equations[3][4].set_color(COLORS[2])

		nng = Group(adj, vec)
		equations[0].move_to(nng)
		equations[0].shift(UP)
		equations[1].next_to(equations[0], DOWN)
		equations[2].next_to(equations[1], DOWN)
		equations[3].next_to(equations[2], DOWN)
		equations_group = Group(*equations)

		self.play(FadeOut(nng), FadeIn(equations_group))
		self.wait(2)
