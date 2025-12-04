import numpy as np
from dendrite import Parameters, Synapse, DendriticSegment

def test_initialization():
    params = Parameters()
    syn = Synapse(syn_type=1, position=0.5, weight=0.2, spike_times=[10.0])
    seg = DendriticSegment.from_params(params, [syn])
    assert seg.psi.shape[0] > 0
    assert seg.psi.shape[1] > 0
