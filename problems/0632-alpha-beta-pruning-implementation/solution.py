import numpy as np

def alpha_beta_pruning(tree, is_maximizing=True):
	"""
	Perform alpha-beta pruning on a game tree.

	Args:
		tree: A nested list representing the game tree. Leaf nodes are numbers,
		      internal nodes are lists of children.
		is_maximizing: Whether the root node is a maximizing player.

	Returns:
		A tuple (value, nodes_evaluated) where value is the optimal minimax
		value and nodes_evaluated is the number of leaf nodes examined.
	"""

	def sub_pruning(tree, is_maximizing, alpha=-np.inf, beta=np.inf):
		
		
		if not isinstance(tree, list):
			return tree, 1

		nodes_evaluated = 0

		if is_maximizing:
			value = -np.inf
			for node in tree:
				node_value, sub_nodes_evaluated = sub_pruning(node, is_maximizing=False, alpha=alpha, beta=beta)
				nodes_evaluated += sub_nodes_evaluated
				value = max(value, node_value)
				alpha = max(alpha, value)
				
				if alpha >= beta:
					break
		else:
			value = np.inf
			for node in tree:
				node_value, sub_nodes_evaluated = sub_pruning(node, is_maximizing=True, alpha=alpha, beta=beta)
				nodes_evaluated += sub_nodes_evaluated
				value = min(value, node_value)
				beta = min(beta, value)

				if alpha >= beta:
					break

		return value, nodes_evaluated
	
	return sub_pruning(tree, is_maximizing=is_maximizing)


