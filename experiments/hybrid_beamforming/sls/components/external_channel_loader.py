import tensorflow as tf
import zarr
import numpy as np


class ExternalChannelLoader:
    """
    Loads channel impulse response (CIR) parameters (a, tau) from preprocessed Zarr stores.
    Bypasses standard geometry-based channel models.
    """

    def __init__(self, zarr_path):
        self.z_root = zarr.open(zarr_path, mode="r")
        self.mesh_points = tf.constant(self.z_root["mesh_points"][:], dtype=tf.float32)
        self.mesh_ids = tf.constant(self.z_root["mesh_ids"][:], dtype=tf.int32)
        self.power_map = self.z_root["power_map"]

    def find_nearest_mesh(self, ut_loc):
        """
        Finds the nearest mesh index for each UT location.
        ut_loc: [batch, num_ut, 3]
        returns: mesh_indices [batch, num_ut] (indices into mesh_ids/points)
        """
        # Distances: [B, N, M]
        ut_loc_exp = tf.expand_dims(ut_loc, axis=2)  # [B, N, 1, 3]
        mesh_exp = tf.expand_dims(
            tf.expand_dims(self.mesh_points, axis=0), axis=0
        )  # [1, 1, M, 3]
        dist = tf.norm(ut_loc_exp - mesh_exp, axis=-1)  # [B, N, M]

        mesh_indices = tf.argmin(dist, axis=-1)  # [B, N]
        return mesh_indices

    def get_power_map(self, ut_mesh_indices):
        """
        Returns the power map for the given UT mesh indices.
        ut_mesh_indices: [batch, num_ut]
        Returns: powers [batch, num_ut, num_sectors] in dBm
        """
        # Zarr indexing can be done with mesh indices
        # power_map is [num_mesh, num_sectors]
        indices = ut_mesh_indices.numpy()
        powers = self.power_map.get_orthogonal_selection(
            (indices.flatten(), slice(None))
        )
        powers = powers.reshape(ut_mesh_indices.shape + (-1,))
        return tf.constant(powers, dtype=tf.float32)

    def get_paths(self, batch_size, neighbor_indices, ut_mesh_indices):
        """
        Returns (a, tau) for the specified neighbor links.
        neighbor_indices: [batch, num_ut, num_neighbors] (BS indices)
        ut_mesh_indices: [batch, num_ut] (indices into mesh_ids)
        Returns:
            a: [batch, num_ut, num_neighbors, num_paths]
            tau: [batch, num_ut, num_neighbors, num_paths]
        """
        num_ut = neighbor_indices.shape[1]
        num_neighbors = neighbor_indices.shape[2]

        all_a = []
        all_tau = []

        # We handle batching in Python for Zarr access (not ideal for performance,
        # but required for 900 sectors sparse access).
        for b in range(batch_size):
            b_a = []
            b_tau = []
            for u in range(num_ut):
                m_idx = ut_mesh_indices[b, u]
                m_id = int(self.mesh_ids[m_idx])

                u_a = []
                u_tau = []
                for k in range(num_neighbors):
                    s_idx = int(neighbor_indices[b, u, k])

                    # Fetch from Zarr: paths/sector_XXX
                    sector_name = f"sector_{s_idx:03d}"
                    try:
                        s_grp = self.z_root["paths"][sector_name]
                        # Find mesh in this sector group
                        s_m_ids = s_grp["mesh_ids"][:]
                        match = np.where(s_m_ids == m_id)[0]

                        if len(match) > 0:
                            p_idx = match[0]
                            # [num_paths]
                            a = s_grp["gains"][p_idx]
                            tau = s_grp["delays"][p_idx]
                        else:
                            # Default empty path
                            a = np.zeros(1, dtype=np.complex64)
                            tau = np.zeros(1, dtype=np.float32)
                    except KeyError:
                        a = np.zeros(1, dtype=np.complex64)
                        tau = np.zeros(1, dtype=np.float32)

                    u_a.append(a)
                    u_tau.append(tau)
                b_a.append(u_a)
                b_tau.append(u_tau)
            all_a.append(b_a)
            all_tau.append(b_tau)

        # Homogenize path counts (Pad to max L)
        max_l = 0
        for b in all_a:
            for u in b:
                for k in u:
                    max_l = max(max_l, len(k))

        a_padded = np.zeros(
            (batch_size, num_ut, num_neighbors, max_l), dtype=np.complex64
        )
        tau_padded = np.zeros(
            (batch_size, num_ut, num_neighbors, max_l), dtype=np.float32
        )

        for b in range(batch_size):
            for u in range(num_ut):
                for k in range(num_neighbors):
                    l_size = len(all_a[b][u][k])
                    a_padded[b, u, k, :l_size] = all_a[b][u][k]
                    tau_padded[b, u, k, :l_size] = all_tau[b][u][k]

        return tf.constant(a_padded), tf.constant(tau_padded)
