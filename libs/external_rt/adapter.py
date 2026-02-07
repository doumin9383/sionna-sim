#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseAdapter(ABC):
    """
    Abstract base class for adapting external HDF5 data schemas to a standard format.
    """

    @abstractmethod
    def map_keys(self, h5_file: Any) -> Dict[str, str]:
        """
        Maps the internal HDF5 keys to the standard schema keys.

        Returns:
            Dict[str, str]: A dictionary mapping standard keys to HDF5 internal paths.
                            Standard keys: 'path_gain', 'delay', 'zenith_at_tx', 'azimuth_at_tx',
                                           'zenith_at_rx', 'azimuth_at_rx', 'velocity_tx', 'velocity_rx'
        """
        pass

    @property
    @abstractmethod
    def required_keys(self) -> List[str]:
        """
        Returns the list of required standard keys.
        """
        pass


class StandardAdapter(BaseAdapter):
    """
    A standard adapter implementation that looks for common naming conventions.
    """

    def __init__(self, key_mapping: Dict[str, str] = None):
        """
        Args:
            key_mapping (Dict[str, str], optional): Custom mapping from standard keys to HDF5 paths.
                                                    If provided, overrides default lookups.
        """
        self.custom_mapping = key_mapping or {}

        # Default potential names for each standard key
        self.default_lookups = {
            "path_gain": ["path_gain", "path_loss", "gain", "power"],
            "delay": ["delay", "tau", "time_of_arrival"],
            "zenith_at_tx": ["zenith_at_tx", "theta_t", "aod_theta"],
            "azimuth_at_tx": ["azimuth_at_tx", "phi_t", "aod_phi"],
            "zenith_at_rx": ["zenith_at_rx", "theta_r", "aoa_theta"],
            "azimuth_at_rx": ["azimuth_at_rx", "phi_r", "aoa_phi"],
            # Velocities might be optional or in specific locations
            "velocity_tx": ["velocity_tx", "v_tx", "tx_vel"],
            "velocity_rx": ["velocity_rx", "v_rx", "rx_vel"],
        }

    @property
    def required_keys(self) -> List[str]:
        return [
            "path_gain",
            "delay",
            "zenith_at_tx",
            "azimuth_at_tx",
            "zenith_at_rx",
            "azimuth_at_rx",
        ]

    def map_keys(self, h5_file: Any) -> Dict[str, str]:
        mapping = {}

        # Helper to find a key in the h5 group/file
        def find_key(possible_names, group):
            for name in possible_names:
                if name in group:
                    return name
            return None

        # 1. Use custom mapping first
        for std_key, h5_path in self.custom_mapping.items():
            if h5_path in h5_file:
                mapping[std_key] = h5_path
            else:
                # Warning could be logged here
                pass

        # 2. Look for missing required keys using defaults
        for std_key in self.required_keys:
            if std_key not in mapping:
                found_path = find_key(self.default_lookups.get(std_key, []), h5_file)
                if found_path:
                    mapping[std_key] = found_path
                else:
                    raise KeyError(
                        f"Could not find a match for required key '{std_key}' in the HDF5 file."
                    )

        # 3. Look for optional keys
        for std_key in ["velocity_tx", "velocity_rx"]:
            if std_key not in mapping:
                found_path = find_key(self.default_lookups.get(std_key, []), h5_file)
                if found_path:
                    mapping[std_key] = found_path

        return mapping
