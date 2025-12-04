# dendritic_segment_model
Dendritic Segment Modelisation

# Methodology

We consider a dendritic segment of length \(L\), characterized by a membrane capacitance \(C_{\text{m}}\) (expressed in \(\si{\milli\second\siemens\per\square\meter}\)). Time is measured in milliseconds. Along this dendritic segment, we use a normalized coordinate system, where the position is given by \(x \in [0,1]\). The distance between position \(x\) and the origin (e.g., soma or another dendritic segment) is \(x \cdot L\).

---

## General Diffusion Model

We consider the following general equation for the membrane potential \(V_{\text{m}}(x,t)\) at time \(t\):



\[
C_{\text{m}} \cdot \frac{\partial V_{\text{m}}}{\partial t}(x,t) =
C_{\text{m}} \cdot D_L \cdot \frac{\partial^2 V_{\text{m}}}{\partial x^2}(x,t)
- I_{\text{leak}}(V_{\text{m}}(x,t))
- I_{\text{syn}}(V_{\text{m}}(x,t), x, t)
- I_{\text{ions}}(V_{\text{m}}(x,t), t)
\]



Where:

- **Diffusion coefficient**:  
  \(D_L = D/L^2\), with \(D\) the diffusion coefficient (\(\si{\meter\squared\per\milli\second}\)).

- **Leak current**:  
  

\[
  I_{\text{leak}}(V) = g_{\text{leak}} \cdot (V - E_{\text{leak}}) = \frac{C_{\text{m}}}{\tau} \cdot (V - E_{\text{leak}})
  \]

  
  with \(g_{\text{leak}}\) the leak conductance, \(E_{\text{leak}}\) the reversal potential (typically \(-65 \, \text{mV}\)), and \(\tau\) the membrane time constant.

- **Synaptic current**:  
  

\[
  I_{\text{syn}}(V, x, t) = \sum_i I_{\text{syn,i}}(V, x, t)
  \]

  
  For a synapse \(i\) at position \(x_i\):  
  

\[
  I_{\text{syn,i}}(V, x, t) = g_{\text{syn,i}}(t) \cdot (V - E_{\text{rev,i}}) \cdot \delta_{x_i}(x)
  \]

  
  - \(E_{\text{rev,i}} = 0 \, \text{mV}\) for excitatory synapses, \(-70 \, \text{mV}\) for inhibitory synapses.  
  - \(g_{\text{syn,i}}(t) = w_i(t) \cdot \overline{g}_i \cdot s_i(t)\), where:
    - \(w_i(t)\): synaptic weight (plasticity-dependent).  
    - \(\overline{g}_i\): maximal conductance.  
    - \(s_i(t)\): normalized synaptic activation trace.  

- **Active ionic currents**:  
  \(I_{\text{ions}}(V, t)\) represent currents from active ion channels (Na\(^+\), K\(^+\), Ca\(^{2+}\), HCN). In this work, we focus on passive dendritic properties and assume subthreshold conditions, so these currents are neglected.

---

## Reduced Membrane Potential

We define the reduced potential:



\[
\psi(x,t) = -\frac{V_{\text{m}}(x,t)-E_{\text{leak}}}{E_{\text{leak}}}
\]



Thus, \(\psi > 0\) indicates depolarization, and \(\psi < 0\) hyperpolarization.

The passive equation becomes:



\[
\frac{\partial \psi}{\partial t}(x,t) =
D_L \frac{\partial^2 \psi}{\partial x^2}(x,t)
- \frac{\psi(x,t)}{\tau}
+ \sum_i \frac{w_i s_i(t)}{\overline{\tau}_i} \cdot (e_i - \psi(x,t)) \cdot \delta_{x_i}(x)
\]



Where:
- \(\overline{\tau}_i = C_{\text{m}}/\overline{g}_i\)  
- \(e_i = (E_{\text{leak}} - E_{\text{rev,i}})/E_{\text{leak}}\)  
- Numerically, \(e_i \approx 1\) for excitatory synapses, \(e_i \approx 0\) for inhibitory synapses.

---

## Spectral Projection Solution

We solve the diffusion equation using Fourier transform and Green’s function methods.  

The solution can be expressed as:



\[
\psi(x,t) = A_0(x,t) + \Psi(x,t) + \Phi(\psi)(x,t)
\]



- **Transient term** \(A_0(x,t)\): depends only on diffusion parameters and initial conditions.  
- **Synaptic term** \(\Psi(x,t)\): convolution of synaptic inputs with the Green’s function.  
- **Boundary term** \(\Phi(\psi)(x,t)\): memory of current flux at segment boundaries, relevant for multi-compartment models.

For long times (\(t \gg \tau\)), the transient term \(A_0\) is neglected, leaving only synaptic contributions.

---

## Numerical Simulations

For numerical simulations, we use a discrete time step \(\delta t = 0.01 \, \text{ms}\).  

We compute recursively:



\[
\Psi_{n+1}(x) = \sum_i \frac{\delta t}{\overline{\tau}_i}
\sum_{p=0}^{n} w_i s_i\!\left((p+\tfrac{1}{2}) \delta t\right) \cdot \left(e_i - \Psi_p(x_i)\right) \cdot G\!\left(x-x_i, \delta t \cdot (n+\tfrac{1}{2}-p)\right)
\]



Thus, \(\Psi(x,t) \approx \Psi_{\lfloor t/\delta t \rfloor}(x)\) as \(\delta t \to 0\).
