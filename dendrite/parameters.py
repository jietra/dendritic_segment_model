from dataclasses import dataclass

@dataclass
class Parameters:
    """
    Simulation parameters for dendritic segment model.

    Attributes
    ----------
    g0 : float
        Baseline conductance density (S/m²). (Note: conductance is around several pS per synapse; here we consider density over areas of around 1µm²)
    Cm : float
        Membrane capacitance (ms·S/m²).
    tau : float
        Membrane time constant (ms).
    tau_rise : float
        Synaptic rise time constant (ms).
    tau_decay : float
        Synaptic decay time constant (ms).
    D : float
        Diffusion coefficient along dendrite (m²/ms). (Note: d/(4*R_i*Cm) in cable model)
    E_leak : float
        Leak reversal potential (mV).
    E_rev_e : float
        Excitatory reversal potential (mV).
    E_rev_i : float
        Inhibitory reversal potential (mV).
    dt : float
        Simulation time step (ms).
    T : float
        Total simulation time (ms).
    dx : float
        Spatial discretization step (normalized length).
    L : float
        Physical length of the dendritic segment (m).
    """

    # Synaptic and membrane parameters
    g0: float = 1.0
    Cm: float = 0.01e3
    tau: float = 10.0
    tau_rise: float = 2.0
    tau_decay: float = 10.0
    D: float = 1.67e-5 / 1e3
    E_leak: float = -65.0
    E_rev_e: float = 0.0
    E_rev_i: float = -70.0

    # Simulation parameters
    dt: float = 0.01
    T: float = 350.0
    dx: float = 0.01
    L: float = 1e-3

    def to_dict(self):
        """Export parameters as dictionary."""
        return self.__dict__

    @staticmethod
    def from_dict(data: dict):
        """Create Parameters instance from dictionary."""
        return Parameters(**data)
