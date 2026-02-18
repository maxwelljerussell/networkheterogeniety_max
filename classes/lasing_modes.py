import numpy as np
from classes.netsalt_config import NetsaltConfig
from classes.pattern import Pattern
from classes.spectrum import Spectrum


class LasingModes:
    def __init__(
        self,
        config: NetsaltConfig,
        pattern: Pattern,
        pump,
        refraction_index,
        quantum_graph_with_pump,
        threshold_modes,
        competition_matrix,
        lasing_modes,
        D0
    ):
        self.config = config
        self.pattern = pattern
        self.pump = pump
        self.refraction_index = refraction_index
        self.quantum_graph_with_pump = quantum_graph_with_pump
        self.threshold_modes = threshold_modes
        self.competition_matrix = competition_matrix
        self.lasing_modes = lasing_modes
        self.D0 = D0

    def get_spectrum(
        self,
        only_lasing_modes: bool = False,
    ) -> Spectrum:
        lasing_modes = self.lasing_modes
        indices = np.arange(len(lasing_modes))

        # Extract frequencies and intensities
        frequencies = np.real(lasing_modes["threshold_lasing_modes"].to_numpy())
        intensities = lasing_modes[lasing_modes.keys()[-1]].to_numpy()
        
        # Filter out non-lasing modes
        if only_lasing_modes:
            good_indices = np.where(~np.isnan(intensities))
            frequencies = frequencies[good_indices]
            intensities = intensities[good_indices]
            indices = indices[good_indices]
        else:
            # Replace NaN values with zeros
            intensities = np.nan_to_num(intensities)

        # Sort frequencies and intensities in ascending order of frequency
        sort_indices = np.argsort(frequencies)
        frequencies = frequencies[sort_indices]
        intensities = intensities[sort_indices]

        return Spectrum(frequencies, intensities, indices, only_lasing_modes)

    def get_mode_indices_in_descending_order_of_intensity(self):
        spectrum = self.get_spectrum(only_lasing_modes=True)
        return spectrum.get_mode_indices_in_descending_order_of_intensity()
