# Benchmark datasets

This archive contains the nine benchmark datasets used to evaluate the performance of EvoMOBO.

**DBD (five datasets).** These datasets describe the binding ability of an ancestral steroid-receptor DNA-binding domain to estrogen response elements (ERE) and steroid response elements (SRE)[1]. For EvoMOBO evaluation, we configured the two-objective benchmarks to maximize SRE binding and minimize ERE binding, and the single-objective benchmark to maximize the difference obtained by subtracting the ERE binding score from the SRE binding score. Mutable positions are E25, G26, K28, and A29. The four two-objective datasets start with 5, 10, 20, or 38 labeled sequences, and the single-objective dataset starts with 38. The original data come from the authors' GitHub repository (https://github.com/JoeThorntonLab/nature-2017_RH-scanning).

**ParD3_1_objectives and ParD3_2_objectives.** The two properties are ParD3 fitness against the toxins ParE2 and ParE3. For EvoMOBO evaluation[2]. we configured the two-objective benchmark to maximize ParE2 fitness and minimize ParE3 fitness, and the single-objective benchmark to maximize the difference obtained by subtracting ParE3 fitness from ParE2 fitness. The fitness data come from the NCBI GEO repository (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153897).

**ParD3_3_objectives.** This dataset contains ParD3 fitness measurements against wild-type ParE2, wild-type ParE3, and the ParE3 toxin carrying V5L/A66F[2-3]. For EvoMOBO evaluation, we configured the optimization to maximize fitness against ParE2 and minimize fitness against both wild-type ParE3 and its V5L/A66F variant. Mutable positions in ParD3 are D61, K64, and E80. The V5L/A66F variant-toxin data come from the authors' GitHub repository (https://github.com/ddingding/coevolution_mechanism/blob/main/data/1raw_read_output/at_combo_lib/df_V5L_A66F.csv).

**ParD3_2_objectives_scale_disparity.** The two properties are ParD3 fitness against ParE2 and the computationally estimated change in folding free energy (ΔΔG). For EvoMOBO evaluation, we configured the optimization to maximize ParE2 fitness and minimize ΔΔG. Mutable positions are D61, K64, and E80. The ΔΔG values are provided in the MosPro dataset[4]. The paired data come from the MosPro Zenodo repository (https://zenodo.org/records/14884203).

## References

[1] T. N. Starr, L. K. Picton, and J. W. Thornton, “Alternative Evolutionary Histories in the Sequence Space of an Ancient Protein,” *Nature* **549** (2017): 409–413. https://doi.org/10.1038/nature23902

[2] T.-L. V. Lite, R. A. Grant, I. Nocedal, M. L. Littlehale, M. S. Guo, and M. T. Laub, “Uncovering the Basis of Protein-Protein Interaction Specificity with a Combinatorially Complete Library,” *eLife* **9** (2020): e60924. https://doi.org/10.7554/eLife.60924

[3] D. Ding, A. G. Green, B. Wang, et al., “Co-evolution of Interacting Proteins through Non-contacting and Non-specific Mutations,” *Nature Ecology & Evolution* **6** (2022): 590–603. https://doi.org/10.1038/s41559-022-01688-0

[4] J. Luo, K. Ding, and Y. Luo, “Pareto-Optimal Sampling for Multi-Objective Protein Sequence Design,” *iScience* **28** (2025): 112119. https://doi.org/10.1016/j.isci.2025.112119
