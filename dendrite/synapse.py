from dataclasses import dataclass

@dataclass
class Synapse:
    """
    Representation of a synapse located on a dendritic segment.

    Attributes
    ----------
    syn_type : int
        Type of synapse (1 for excitatory, 0 for inhibitory).
    position : float
        Position of the synapse along the dendrite (normalized coordinate in [0,1]).
    weight : float
        Synaptic weight (scaling factor for conductance).
    spike_times : list of float
        List of spike times (ms) received at this synapse.
    """

    syn_type: int
    position: float
    weight: float
    spike_times: list

    def to_dict(self):
        """Export synapse attributes as dictionary."""
        return {
            'syn_type': self.syn_type,
            'position': self.position,
            'weight': self.weight,
            'spike_times': self.spike_times
        }

    @staticmethod
    def from_dict(data: dict):
        """Create Synapse instance from dictionary."""
        return Synapse(
            syn_type=data['syn_type'],
            position=data['position'],
            weight=data['weight'],
            spike_times=data['spike_times']
        )
