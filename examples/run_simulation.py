import matplotlib.pyplot as plt
from dendrite import Parameters, Synapse, DendriticSegment

def main():
    # 1. Define simulation parameters
    params = Parameters(
        T=50.0,     # total simulation time (ms)
        dt=0.1,     # time step (ms)
        dx=0.01,    # spatial discretization step (normalized length)
        L=1e-3      # dendritic segment length (m)
    )

    # 2. Create a simple excitatory synapse located in the middle of the dendrite
    syn = Synapse(
        syn_type=1,          # 1 = excitatory synapse
        position=0.5,        # normalized position along the dendrite
        weight=0.2,          # synaptic weight
        spike_times=[10.0]   # one spike occurring at 10 ms
    )

    # 3. Initialize the dendritic segment with parameters and synapse
    segment = DendriticSegment.from_params(params, [syn])

    # 4. Run the simulation
    segment.simulate()

    # 5. Print a description of the segment and its parameters
    print(segment.describe())

    # 6. Plot the membrane potential at the center of the dendrite
    center_index = len(segment.x_vals) // 2
    plt.plot(segment.t_vals, segment.psi[center_index, :], label="Center potential")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (a.u.)")
    plt.title("Membrane potential at dendrite center")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
