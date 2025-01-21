# Mongoose

## Abstract

<div align='justify'> Perturbation modelling in single-cell data is crucial for studying molecular changes elicited due to molecular knockouts, chemical compounds, and biological stimulants across health and disease phenotypes. Perturbation modelling is confounded by scarce biological explainability, statistical uncertainty in Deep Learning (DL) predictions in extreme perturbation scenarios, hyperparameter optimization, and limited scalability to multi-omic single-cell data. We present the Mongoose project (Multi-Objective Network Generator Of Optimized Single-cell Experiments) to explore enhancing perturbation modelling in complex single-cell datasets through Multi-Task Learning (MTL). Mongoose combines (i) UnitedNet, a DL framework that employs MTL to simultaneously perform joint group identification and cross-modal prediction with Sharpley values (SHAP) as an explainability component and (ii) perturbation modelling tools like SCING and GenKI, which reconstruct and perturb cell type-specific gene-regulatory networks (GRNs). UnitedNet contains an encoder-decoder-discriminator structure which approximates the statistical characteristics of each modality without prior assumptions. Hence, we claim that UnitedNet can facilitate biologically informed decisions on conducting ensuing digital KOs across reverse-engineered GRNs by providing (i) better cell-type clusters and (ii) SHAP values of significant cross-modal/cell-type associations from complex single-cell and spatial omic datasets. We will showcase the Mongoose approach on complex multi-omic mRNA/protein datasets like the Perturb-CITE-seq (CRISPR knock-outs) and the spatial DBiT-seq mouse embryo dataset, where mRNA-protein associations and spatial niche identification are expected to play pivotal roles in perturbations like in-silico GRN digital KOs. We anticipate that Mongoose will provide perturbational insights closer to ground truths, ultimately highlighting critical transcription factors and signalling pathways with potential translational value. </div>



<br><br>
![ECCB2024](https://raw.githubusercontent.com/BiodataAnalysisGroup/Mongoose/main/Images/ECCB_2024.png)



## Contribute

We welcome and greatly appreciate any sort of feedback and/or contribution!

If you have any questions, please either open an issue or write us at `ggeorav@certh.gr` or `inab.bioinformatics@lists.certh.gr`.
