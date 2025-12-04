import numpy as np
import json
import copy
from datetime import datetime

from .parameters import Parameters
from .synapse import Synapse
from .simulation import psiu

class DendriticSegment:
    """
    Computational model of a dendritic segment with synaptic inputs.

    This class simulates the membrane potential dynamics along a dendrite
    in response to synaptic inputs, using Green's function methods.

    Attributes
    ----------
    psi : np.ndarray
        Membrane potential array over space and time.
    x_vals : np.ndarray
        Spatial discretization points along the dendrite.
    t_vals : np.ndarray
        Temporal discretization points for the simulation.
    syn_weights : np.ndarray
        Weights of the synapses.
    syn_positions : np.ndarray
        Positions of the synapses along the dendrite.
    spike_times_list : list of np.ndarray
        List of spike times for each synapse.
    syn_types : list of int
        Types of synapses (1 for excitatory, 0 for inhibitory).
    e_r : np.ndarray
        Reversal potentials normalized by leak potential.
    params : Parameters
        Simulation parameters.
    synapses : list of Synapse
        List of synaptic inputs.

    
    Methods
    -------
    simulate(save_path=None)
        Runs the simulation and optionally saves the results.
    save_results(filename=None)
        Saves the simulation results to a .npz file.
    load_from_npz(filename)
        Loads simulation data from a .npz file.
    from_file(cls, filename)
        Class method to create an instance from a .npz file.
    from_params(cls, params, synapses)
        Class method to create an instance from parameters and synapses.
    from_config_file(cls, json_path)
        Class method to create an instance from a JSON configuration file.
    export_config()
        Exports the current configuration to a dictionary.
    save_config(filename='dendrite_config.json')
        Saves the current configuration to a JSON file.
    describe()
        Returns a string description of the dendritic segment and its parameters.
    """

    def __init__(self, params: Parameters = None, synapses: list = None, npz_file: str = None):
        if npz_file:
            self.load_from_npz(npz_file)
        elif params and synapses:
            self.initialize(params, synapses)
        else:
            self.initialize(Parameters(), [])

    @classmethod
    def from_file(cls, filename: str):
        instance = cls()
        instance.load_from_npz(filename)
        return instance
    
    @classmethod
    def from_params(cls, params: Parameters, synapses: list):
        instance = cls()
        instance.initialize(params, synapses)
        return instance
    
    @classmethod
    def from_config_file(cls, json_path: str):
        with open(json_path, 'r') as f:
            config = json.load(f)
        params   = Parameters.from_dict(config['params'])
        synapses = [Synapse.from_dict(syn_dict) for syn_dict in config['synapses']]
        return cls.from_params(params, synapses)

    def initialize(self, params: Parameters, synapses: list):
        self.params   = copy.deepcopy(params)
        self.synapses = synapses

        self.t_vals = np.arange(0, self.params.T + self.params.dt, self.params.dt, dtype=np.float32)
        self.x_vals = np.arange(0, 1 + self.params.dx, self.params.dx, dtype=np.float32)
    
        self.psi = np.zeros((len(self.x_vals), len(self.t_vals)), dtype=np.float32)
        
        # Precompute synapse-related arrays
        self.syn_weights      = np.array([s.weight for s in self.synapses], dtype=np.float32)
        self.syn_positions    = np.array([s.position for s in self.synapses], dtype=np.float32)
        self.syn_indices      = np.searchsorted(self.x_vals, self.syn_positions)
        self.spike_times_list = [np.array(s.spike_times, dtype=np.float32) for s in self.synapses]
        self.syn_types        = [s.syn_type for s in self.synapses]

        self.e_r = np.empty(len(self.synapses), dtype=np.float32)
        for i, s in enumerate(self.synapses):
            E_rev = self.params.E_rev_e if s.syn_type == 1 else self.params.E_rev_i
            self.e_r[i] = 1.0 - (E_rev / self.params.E_leak)

    def simulate(self, save_path: str = None):
        D_L      = np.float32(self.params.D / (self.params.L ** 2))
        tau      = np.float32(self.params.tau)
        tau_rise = np.float32(self.params.tau_rise)
        tau_decay= np.float32(self.params.tau_decay)
        dt       = np.float32(self.params.dt)
        tau_i    = np.float32(self.params.Cm / self.params.g0)

        for n in range(len(self.t_vals) - 1):
            for i in range(len(self.synapses)):
                self.psi[:, n + 1] += self.syn_weights[i] * psiu(
                    self.psi, n, dt, self.spike_times_list[i],
                    tau_rise, tau_decay, D_L, tau,
                    self.x_vals, self.syn_positions[i], self.e_r[i]
                )
            self.psi[:, n + 1] = self.psi[:, n + 1] / tau_i

        if save_path is not None:
            self.save_results(save_path)

    def save_results(self, filename: str = None):
        if filename is None:
            dx_str    = f"{self.params.dx:.3f}".replace('.', 'p')
            L_str     = f"{int(self.params.L * 1e6)}"
            T_str     = f"{int(self.params.T)}"
            n_syn     = len(self.synapses)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"res_dx{dx_str}_L{L_str}microns_T{T_str}ms_{n_syn}synapses_{timestamp}.npz"

        np.savez(filename,
            psi             = self.psi,
            x_vals          = self.x_vals,
            t_vals          = self.t_vals,
            syn_weights     = self.syn_weights,
            syn_positions   = self.syn_positions,
            syn_indices     = self.syn_indices,
            spike_times_list= self.spike_times_list,
            syn_types       = self.syn_types,
            params_L        = self.params.L,
            params_dx       = self.params.dx,
            params_dt       = self.params.dt,
            params_T        = self.params.T,
            params_g0       = self.params.g0,
            params_Cm       = self.params.Cm,
            params_tau      = self.params.tau,
            params_tau_rise = self.params.tau_rise,
            params_tau_decay= self.params.tau_decay,
            params_D        = self.params.D,
            params_E_leak   = self.params.E_leak,
            params_E_rev_e  = self.params.E_rev_e,
            params_E_rev_i  = self.params.E_rev_i
        )

    def load_from_npz(self, filename):
        data = np.load(filename, allow_pickle=True)

        self.psi                = data['psi']
        self.x_vals             = data['x_vals']
        self.t_vals             = data['t_vals']
        self.syn_weights        = data['syn_weights']
        self.syn_positions      = data['syn_positions']
        self.syn_indices        = data.get('syn_indices', np.searchsorted(self.x_vals, self.syn_positions))
        self.spike_times_list   = data['spike_times_list']
        self.syn_types          = data['syn_types']

        self.params = Parameters(
            L   = data['params_L'].item(),
            dx  = data.get('params_dx', self.x_vals[1] - self.x_vals[0]).item(),
            dt  = data.get('params_dt', self.t_vals[1] - self.t_vals[0]).item(),
            T   = data.get('params_T', self.t_vals[-1]).item(),
            g0          = data.get('params_g0', 0.5).item(),
            Cm          = data.get('params_Cm', 0.01e3).item(),
            tau         = data.get('params_tau', 10).item(),
            tau_rise    = data.get('params_tau_rise', 2).item(),
            tau_decay   = data.get('params_tau_decay', 10).item(),
            D           = data.get('params_D', (200e-6)**2/1e3).item(),
            E_leak      = data.get('params_E_leak', -65.0).item(),
            E_rev_e     = data.get('params_E_rev_e', 0).item(),
            E_rev_i     = data.get('params_E_rev_i', -70.0).item()
        )
        self.synapses = [
            Synapse(position=pos, weight=w, syn_type=stype, spike_times=spikes.tolist())
            for pos, w, stype, spikes in zip(self.syn_positions, self.syn_weights, self.syn_types, self.spike_times_list)
        ]
        self.e_r = 1 - np.array([self.params.E_rev_e if s.syn_type == 1 else self.params.E_rev_i for s in self.synapses]) / self.params.E_leak

    def export_config(self):
        return {
            'params': self.params.to_dict(),
            'synapses': [s.to_dict() for s in self.synapses]
        }
    
    def save_config(self, filename: str = 'dendrite_config.json'):
        config = self.export_config()
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)

    def describe(self):
        desc    = f"DendriticSegment with {len(self.synapses)} synapses\n"
        desc    += f"  Length (L): {self.params.L * 1e6:.1f} µm\n"
        desc    += f"  Spatial step (dx): {self.params.dx * self.params.L * 1e6:.3f} µm\n"
        desc    += f"  Temporal step (dt): {self.params.dt:.3f} ms\n"
        desc    += f"  Total time (T): {self.params.T:.1f} ms\n"
        desc    += f"Parameters:\n"
        for key, value in self.params.to_dict().items():
            desc += f"  {key}: {value}\n"
        desc    += "Synapses: {len(self.synapses)}\n"
        for i, syn in enumerate(self.synapses):
            type_str = 'Excitatory' if syn.syn_type == 1 else 'Inhibitory'
            desc += f"  Synapse {i+1}: {type_str}\n"
            for key, value in syn.to_dict().items():
                desc += f"    {key}: {value}\n"
        return desc
