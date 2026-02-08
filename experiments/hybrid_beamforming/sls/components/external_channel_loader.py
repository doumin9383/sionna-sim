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
        indices = ut_mesh_indices.numpy()
        b, n = indices.shape
        flat_indices = indices.flatten()

        # Explicitly fetch rows to handle duplicates correctly
        # [num_ut_total, num_sectors]
        powers = np.array([self.power_map[idx] for idx in flat_indices])

        # Reshape to [batch, num_ut, num_sectors]
        powers = powers.reshape((b, n, -1))
        return tf.constant(powers, dtype=tf.float32)

    def get_paths(self, batch_size, neighbor_indices, ut_mesh_indices):
        """
        Returns (a, tau, doa_az, doa_el, dod_az, dod_el) for the specified neighbor links.
        """
        num_ut = neighbor_indices.shape[1]
        num_neighbors = neighbor_indices.shape[2]

        all_a, all_tau = [], []
        all_doa_az, all_doa_el = [], []
        all_dod_az, all_dod_el = [], []

        for b in range(batch_size):
            b_a, b_tau = [], []
            b_doa_az, b_doa_el = [], []
            b_dod_az, b_dod_el = [], []
            for u in range(num_ut):
                m_idx = ut_mesh_indices[b, u]
                m_id = int(self.mesh_ids[m_idx])

                u_a, u_tau = [], []
                u_doa_az, u_doa_el = [], []
                u_dod_az, u_dod_el = [], []
                for k in range(num_neighbors):
                    s_idx = int(neighbor_indices[b, u, k])
                    sector_name = f"sector_{s_idx:03d}"
                    try:
                        s_grp = self.z_root["paths"][sector_name]
                        s_m_ids = s_grp["mesh_ids"][:]
                        match = np.where(s_m_ids == m_id)[0]

                        if len(match) > 0:
                            p_idx = match[0]
                            a = s_grp["gains"][p_idx]
                            tau = s_grp["delays"][p_idx]
                            doa_az = s_grp["doa_az"][p_idx]
                            doa_el = s_grp["doa_el"][p_idx]
                            dod_az = s_grp["dod_az"][p_idx]
                            dod_el = s_grp["dod_el"][p_idx]
                        else:
                            a, tau = np.zeros(1, dtype=np.complex64), np.zeros(
                                1, dtype=np.float32
                            )
                            doa_az, doa_el = np.zeros(1, dtype=np.float32), np.zeros(
                                1, dtype=np.float32
                            )
                            dod_az, dod_el = np.zeros(1, dtype=np.float32), np.zeros(
                                1, dtype=np.float32
                            )
                    except KeyError:
                        a, tau = np.zeros(1, dtype=np.complex64), np.zeros(
                            1, dtype=np.float32
                        )
                        doa_az, doa_el = np.zeros(1, dtype=np.float32), np.zeros(
                            1, dtype=np.float32
                        )
                        dod_az, dod_el = np.zeros(1, dtype=np.float32), np.zeros(
                            1, dtype=np.float32
                        )

                    u_a.append(a)
                    u_tau.append(tau)
                    u_doa_az.append(doa_az)
                    u_doa_el.append(doa_el)
                    u_dod_az.append(dod_az)
                    u_dod_el.append(dod_el)
                b_a.append(u_a)
                b_tau.append(u_tau)
                b_doa_az.append(u_doa_az)
                b_doa_el.append(u_doa_el)
                b_dod_az.append(u_dod_az)
                b_dod_el.append(u_dod_el)
            all_a.append(b_a)
            all_tau.append(b_tau)
            all_doa_az.append(b_doa_az)
            all_doa_el.append(b_doa_el)
            all_dod_az.append(b_dod_az)
            all_dod_el.append(b_dod_el)

        # Pad and Convert to TF
        max_l = max(len(k) for b in all_a for u in b for k in u)

        def pad_and_stack(data, dtype):
            padded = np.zeros((batch_size, num_ut, num_neighbors, max_l), dtype=dtype)
            for b in range(batch_size):
                for u in range(num_ut):
                    for k in range(num_neighbors):
                        l_size = len(data[b][u][k])
                        padded[b, u, k, :l_size] = data[b][u][k]
            return tf.constant(padded)

        a_tf = pad_and_stack(all_a, np.complex64)
        tau_tf = pad_and_stack(all_tau, np.float32)
        doa_az_tf = pad_and_stack(all_doa_az, np.float32)
        doa_el_tf = pad_and_stack(all_doa_el, np.float32)
        dod_az_tf = pad_and_stack(all_dod_az, np.float32)
        dod_el_tf = pad_and_stack(all_dod_el, np.float32)

        # Shape a for Sionna compatibility [B, U, 1, K, 1, L, 1]
        a_tf = a_tf[:, :, tf.newaxis, :, tf.newaxis, :, tf.newaxis]

        return a_tf, tau_tf, doa_az_tf, doa_el_tf, dod_az_tf, dod_el_tf
