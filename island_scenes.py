from manim import *

scale = 0.5

nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
edges = [(1, 2), (2, 4), (4, 7), (7, 8), (7, 9), (1, 9), (2, 9), (3, 5), (5, 6), (3, 6)]
GRAPH = Graph(nodes, edges, layout="circular")
GRAPH.scale(scale)
ADJ_MATRIX = Matrix([
	[0, 1, 0, 0, 0, 0, 0, 0, 1],
	[1, 0, 0, 1, 0, 0, 0, 0, 1],
	[0, 0, 0, 0, 1, 1, 0, 0, 0],
	[0, 1, 0, 0, 0, 0, 1, 0, 0],
	[0, 0, 1, 0, 0, 1, 0, 0, 0],
	[0, 0, 1, 0, 1, 0, 0, 0, 0],
	[0, 0, 0, 1, 0, 0, 0, 1, 1],
	[0, 0, 0, 0, 0, 0, 1, 0, 0],
	[1, 1, 0, 0, 0, 0, 1, 0, 0]
])
ADJ_MATRIX.scale(scale)

class IslandD(Scene):
	def construct(self):
		graph_copy = GRAPH.copy()
		matrix = ADJ_MATRIX.copy()
		matrix.shift(LEFT*3)
		matrix.shift(UP*1.5)
		island_vector = Matrix([[0], [0], [1], [0], [1], [1], [0], [0], [0]], right_bracket=')', left_bracket='(')
		island_vector.scale(scale)
		island_vector.next_to(matrix, RIGHT)
		equals = Tex('=')
		equals.next_to(island_vector, RIGHT)
		product_vector = Matrix([[0], [0], [2], [0], [2], [2], [0], [0], [0]], right_bracket=')', left_bracket='(')
		product_vector.scale(scale)
		product_vector.next_to(equals, RIGHT)
		graph_copy.next_to(matrix, DOWN)

		lamda = MathTex(r'\lambda')
		island_vector_copy = island_vector.copy()
		product_group = Group(matrix, island_vector, equals)
		self.play(FadeIn(matrix), Create(graph_copy))
		self.wait(1)
		self.play(FadeIn(island_vector), FadeIn(equals), FadeIn(product_vector))
		self.wait(1)

		lamda.next_to(equals, RIGHT)
		island_vector_copy.next_to(lamda, RIGHT)
		eigen_group = Group(lamda, island_vector_copy)
		self.play(FadeOut(product_vector, shift=DOWN), FadeIn(eigen_group, shift=DOWN))
		self.wait(2)

		#Highlight island portions in yellow

		coloring_list = [WHITE, WHITE, YELLOW, WHITE, YELLOW, YELLOW, WHITE, WHITE, WHITE]
		self.play(graph_copy[3].animate.set_color(YELLOW),
			graph_copy[5].animate.set_color(YELLOW),
			graph_copy[6].animate.set_color(YELLOW),
			graph_copy.edges[(3, 5)].animate.set_color(YELLOW),
			graph_copy.edges[(5, 6)].animate.set_color(YELLOW),
			graph_copy.edges[(3, 6)].animate.set_color(YELLOW),
			island_vector.animate.set_row_colors(*coloring_list),
			island_vector_copy.animate.set_row_colors(*coloring_list)
		)
		self.wait(2)
