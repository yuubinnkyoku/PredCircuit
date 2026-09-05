# References

Core papers and software relevant to the initial scope.

1. Whittington, J. C. R. & Bogacz, R. (2017). *An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity*. Neural Computation 29(5), 1229–1262. DOI: 10.1162/NECO_a_00949.
2. Millidge, B., Tschantz, A. & Buckley, C. L. (2020). *Predictive Coding Approximates Backprop along Arbitrary Computation Graphs*. arXiv:2006.04182.
3. Millidge, B. et al. (2022). *Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?* arXiv:2202.09467.
4. Millidge, B. et al. (2022). *A Theoretical Framework for Inference and Learning in Predictive Coding Networks*. arXiv:2207.12316.
5. Lappalainen, J. K. et al. (2024). *Connectome-constrained networks predict neural activity across the fly visual system*. Nature 634, 1132–1140. DOI: 10.1038/s41586-024-07939-3. Code: TuragaLab/flyvis.
6. Berg, S. et al. MaleCNS project, dataset `male-cns:v1.0`, official Janelia project/download pages. Dataset license: CC-BY.

## Why these are here

The first four define the predictive-coding/local-learning baseline and its relationship to backpropagation. Lappalainen et al. demonstrates that measured fly connectivity can usefully constrain a task-optimized recurrent model, while still using BPTT. MaleCNS provides the larger biological graph on which PredCircuit can test predictive-coding-specific hypotheses.
