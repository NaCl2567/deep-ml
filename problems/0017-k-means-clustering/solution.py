import  numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here

	point_to_centroid = None
	D = len(points[0])
	points_arr = np.array(points)
	final_centroids_arr = np.array(initial_centroids)
	for _ in range(max_iterations):
		### Assignment Step
		# points: (N, D), centroids: (K, D)
		
		p_sq = np.einsum('ij,ij->i', points_arr, points_arr)        # (N,)
		c_sq = np.einsum('ij,ij->i', final_centroids_arr, final_centroids_arr)  # (K,)

		sq = p_sq[:, None] + c_sq[None, :] - 2.0 * (points_arr @ final_centroids_arr.T)  # (N, K)
		np.maximum(sq, 0.0, out=sq)                          # 防止浮点误差出负数
		
		# all_distances = np.concatenate(all_distances, axis=1) 	# N, k
		point_to_centroid = np.argmin(sq, axis=1)	# N, 1
		# return points[np.where(point_to_centroid == 0)[0]]
		# return type(np.where(point_to_centroid == 0)[0])

		### Update Step
		# for i in range(k):
			
		# 	assigned_points = points[list(np.where(point_to_centroid == i)[0])]	# c, d
		# 	# print(assigned_points)
		# 	if assigned_points.shape[0] != 0:
		# 		final_centroids[i] = tuple(assigned_points.mean(axis=0, keepdims=False))
		counts = np.bincount(point_to_centroid, minlength=k)
		sums = np.empty((k, D), dtype=np.float64)
		for d in range(D):                                      # D 远小于 K
			sums[:, d] = np.bincount(point_to_centroid, weights=points_arr[:, d], minlength=k)

		empty = counts == 0
		# return empty
		new_centroids = np.divide(
			sums, counts[:, None].astype(np.float64),
			out=np.zeros_like(sums),
			where=~empty[:, None],
		)
		# return np.where(counts == 0)[0]
		empty_indices = np.where(counts == 0)[0]
		# empty_indices = np.array([0,2,3])
		new_centroids[empty_indices] = final_centroids_arr[empty_indices] 
		final_centroids_arr = new_centroids
	
	final_centroids = [tuple(final_centroids_arr[i]) for i in range(final_centroids_arr.shape[0])]

	return final_centroids