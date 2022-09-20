# Taken from Reducible's github

from manim import *
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
#from functions import *
from typing import Hashable, Iterable

import numpy as np
import itertools as it

REDUCIBLE_PURPLE=RED
REDUCIBLE_VIOLET=RED



class MarkovChain:
    def __init__(
        self,
        states: int,
        edges: list[tuple[int, int]],
        transition_matrix=None,
        dist=None,
    ):
        """
        @param: states -- number of states in Markov Chain
        @param: edges -- list of tuples (u, v) for a directed edge u to v, u in range(0, states), v in range(0, states)
        @param: transition_matrix -- custom np.ndarray matrix of transition probabilities for all states in Markov chain
        @param: dist -- initial distribution across states, assumed to be uniform if none
        """
        self.states = range(states)
        self.edges = edges
        self.adj_list = {}
        for state in self.states:
            self.adj_list[state] = []
            for u, v in edges:
                if u == state:
                    self.adj_list[state].append(v)

        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            # Assume default transition matrix is uniform across all outgoing edges
            self.transition_matrix = np.zeros((states, states))
            for state in self.states:
                neighbors = self.adj_list[state]
                for neighbor in neighbors:
                    self.transition_matrix[state][neighbor] = 1 / len(neighbors)

        # handle sink nodes to point to itself
        for i, row in enumerate(self.transition_matrix):
            if np.sum(row) == 0:
                self.transition_matrix[i][i] = 1

        if dist is not None:
            self.dist = dist
        else:
            self.dist = np.array(
                [1 / len(self.states) for _ in range(len(self.states))]
            )

        self.starting_dist = self.dist

    def get_states(self):
        return list(self.states)

    def get_edges(self):
        return self.edges

    def get_adjacency_list(self):
        return self.adj_list

    def get_transition_matrix(self):
        return self.transition_matrix

    def get_current_dist(self):
        return self.dist

    def update_dist(self):
        """
        Performs one step of the markov chain
        """
        self.dist = np.dot(self.dist, self.transition_matrix)

    def get_true_stationary_dist(self):
        dist = np.linalg.eig(np.transpose(self.transition_matrix))[1][:, 0]
        return dist / sum(dist)

    def set_starting_dist(self, starting_dist):
        self.starting_dist = starting_dist
        self.dist = starting_dist

    def get_starting_dist(self):
        return self.starting_dist

    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = transition_matrix

class CustomLabel(Text):
    def __init__(self, label, font="SF Mono", scale=1, weight=BOLD):
        super().__init__(label, font=font, weight=weight)
        self.scale(scale)


class CustomCurvedArrow(CurvedArrow):
    def __init__(self, start, end, tip_length=0.15, **kwargs):
        super().__init__(start, end, **kwargs)
        self.pop_tips()
        self.add_tip(
            tip_shape=ArrowTriangleFilledTip,
            tip_length=tip_length,
            at_start=False,
        )
        self.tip.z_index = -100

    def set_opacity(self, opacity, family=True):
        return super().set_opacity(opacity, family)

    @override_animate(set_opacity)
    def _set_opacity_animation(self, opacity=1, anim_args=None):
        if anim_args is None:
            anim_args = {}

        animate_stroke = self.animate.set_stroke(opacity=opacity)
        animate_tip = self.tip.animate.set_opacity(opacity)

        return AnimationGroup(*[animate_stroke, animate_tip])


class MarkovChainGraph(Graph):
    def __init__(
        self,
        markov_chain: MarkovChain,
        vertex_config={
            "stroke_color": REDUCIBLE_PURPLE,
            "stroke_width": 3,
            "fill_color": REDUCIBLE_PURPLE,
            "fill_opacity": 0.5,
        },
        curved_edge_config: dict = None,
        straight_edge_config: dict = None,
        enable_curved_double_arrows=True,
        labels=True,
        state_color_map=None,
        **kwargs,
    ):
        self.markov_chain = markov_chain
        self.enable_curved_double_arrows = enable_curved_double_arrows

        self.default_curved_edge_config = {
            "color": REDUCIBLE_VIOLET,
            "stroke_width": 3,
            "radius": 4,
        }

        self.default_straight_edge_config = {
            "color": REDUCIBLE_VIOLET,
            "max_tip_length_to_length_ratio": 0.06,
            "stroke_width": 3,
        }
        self.state_color_map = state_color_map

        if labels:
            labels = {
                k: CustomLabel(str(k), scale=0.6) for k in markov_chain.get_states()
            }

        if self.state_color_map:
            new_vertex_config = {}
            for state in markov_chain.get_states():
                new_vertex_config[state] = vertex_config.copy()
                new_vertex_config[state]["stroke_color"] = self.state_color_map[state]
                new_vertex_config[state]["fill_color"] = self.state_color_map[state]

            vertex_config = new_vertex_config

        self.labels = {}

        super().__init__(
            markov_chain.get_states(),
            markov_chain.get_edges(),
            vertex_config=vertex_config,
            labels=labels,
            **kwargs,
        )

        self._graph = self._graph.to_directed()
        self.remove_edges(*self.edges)

        self.add_markov_chain_edges(
            *markov_chain.get_edges(),
            straight_edge_config=straight_edge_config,
            curved_edge_config=curved_edge_config,
        )

        self.clear_updaters()
        # this updater makes sure the edges remain connected
        # even when states move around
        def update_edges(graph):
            for (u, v), edge in graph.edges.items():
                v_c = self.vertices[v].get_center()
                u_c = self.vertices[u].get_center()
                vec = v_c - u_c
                unit_vec = vec / np.linalg.norm(vec)

                u_radius = self.vertices[u].width / 2
                v_radius = self.vertices[v].width / 2

                arrow_start = u_c + unit_vec * u_radius
                arrow_end = v_c - unit_vec * v_radius
                edge.put_start_and_end_on(arrow_start, arrow_end)

        self.add_updater(update_edges)
        update_edges(self)

    def add_edge_buff(
        self,
        edge: tuple[Hashable, Hashable],
        edge_type: type[Mobject] = None,
        edge_config: dict = None,
    ):
        """
        Custom function to add edges to our Markov Chain,
        making sure the arrowheads land properly on the states.
        """
        if edge_config is None:
            edge_config = self.default_edge_config.copy()
        added_mobjects = []
        for v in edge:
            if v not in self.vertices:
                added_mobjects.append(self._add_vertex(v))
        u, v = edge

        self._graph.add_edge(u, v)

        base_edge_config = self.default_edge_config.copy()
        base_edge_config.update(edge_config)
        edge_config = base_edge_config
        self._edge_config[(u, v)] = edge_config

        v_c = self.vertices[v].get_center()
        u_c = self.vertices[u].get_center()
        vec = v_c - u_c
        unit_vec = vec / np.linalg.norm(vec)

        if self.enable_curved_double_arrows:
            arrow_start = u_c + unit_vec * self.vertices[u].radius
            arrow_end = v_c - unit_vec * self.vertices[v].radius
        else:
            arrow_start = u_c
            arrow_end = v_c
            edge_config["buff"] = self.vertices[u].radius

        edge_mobject = edge_type(
            start=arrow_start, end=arrow_end, z_index=-100, **edge_config
        )
        self.edges[(u, v)] = edge_mobject

        self.add(edge_mobject)
        added_mobjects.append(edge_mobject)
        return self.get_group_class()(*added_mobjects)

    def add_markov_chain_edges(
        self,
        *edges: tuple[Hashable, Hashable],
        curved_edge_config: dict = None,
        straight_edge_config: dict = None,
        **kwargs,
    ):
        """
        Custom function for our specific case of Markov Chains.
        This function aims to make double arrows curved when two nodes
        point to each other, leaving the other ones straight.
        Parameters
        ----------
        - edges: a list of tuples connecting states of the Markov Chain
        - curved_edge_config: a dictionary specifying the configuration
        for CurvedArrows, if any
        - straight_edge_config: a dictionary specifying the configuration
        for Arrows
        """

        if curved_edge_config is not None:
            curved_config_copy = self.default_curved_edge_config.copy()
            curved_config_copy.update(curved_edge_config)
            curved_edge_config = curved_config_copy
        else:
            curved_edge_config = self.default_curved_edge_config.copy()

        if straight_edge_config is not None:
            straight_config_copy = self.default_straight_edge_config.copy()
            straight_config_copy.update(straight_edge_config)
            straight_edge_config = straight_config_copy
        else:
            straight_edge_config = self.default_straight_edge_config.copy()

        print(straight_edge_config)

        edge_vertices = set(it.chain(*edges))
        new_vertices = [v for v in edge_vertices if v not in self.vertices]
        added_vertices = self.add_vertices(*new_vertices, **kwargs)

        edge_types_dict = {}
        for e in edges:
            if self.enable_curved_double_arrows and (e[1], e[0]) in edges:
                edge_types_dict.update({e: (CustomCurvedArrow, curved_edge_config)})

            else:
                edge_types_dict.update({e: (Arrow, straight_edge_config)})

        added_mobjects = sum(
            (
                self.add_edge_buff(
                    edge,
                    edge_type=e_type_and_config[0],
                    edge_config=e_type_and_config[1],
                ).submobjects
                for edge, e_type_and_config in edge_types_dict.items()
            ),
            added_vertices,
        )

        return self.get_group_class()(*added_mobjects)

    def get_transition_labels(self, scale=0.3, round_val=True):
        """
        This function returns a VGroup with the probability that each
        each state has to transition to another state, based on the
        Chain's transition matrix.
        It essentially takes each edge's probability and creates a label to put
        on top of it, for easier indication and explanation.
        This function returns the labels already set up in a VGroup, ready to just
        be created.
        """
        tm = self.markov_chain.get_transition_matrix()

        labels = VGroup()
        for s in range(len(tm)):

            for e in range(len(tm[0])):
                if s != e and tm[s, e] != 0:

                    edge_tuple = (s, e)
                    matrix_prob = tm[s, e]

                    if round_val and round(matrix_prob, 2) != matrix_prob:
                        matrix_prob = round(matrix_prob, 2)

                    label = (
                        Text(str(matrix_prob))
                        .set_stroke(BLACK, width=8, background=True, opacity=0.8)
                        .scale(scale)
                        .move_to(self.edges[edge_tuple].point_from_proportion(0.2))
                    )

                    labels.add(label)
                    self.labels[edge_tuple] = label

        def update_labels(graph):
            for e, l in graph.labels.items():
                l.move_to(graph.edges[e].point_from_proportion(0.2))

        self.add_updater(update_labels)

        return labels

