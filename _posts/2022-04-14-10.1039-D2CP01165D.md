# Solving real-world problems with kallisto

## Authors

- Eike Caldeweyher
- Christoph Bauer
- Ali Soltani Tehrani

- [DOI:10.1039/D2CP01165D](https://doi.org/10.1039/D2CP01165D)

## Abstract

We present the open-source framework kallisto that enables the efficient and robust calculation of quantum mechanical features for atoms and molecules.
For a benchmark set of 49 experimental molecular polarizabilities, the predictive power of the presented method competes against second-order perturbation theory in a converged atomic-orbital basis set at a fraction of its computational costs.
The calculation of isotropic molecular polarizabilities is robust for a data set of more than 80 000 molecules.
We present furthermore a generally applicable van der Waals radius model that is rooted on atomic static polarizabilites.
Efficiency tests show that such radii can even be calculated for small- to medium-size proteins where the largest system (SARS-CoV-2 spike protein) has 42 539 atoms.
Following the work of Domingo-Alemenara et al. [Domingo-Alemenara et al., Nat. Commun., 2019, 10, 5811], we present computational predictions for retention times for different chromatographic methods and describe how physicochemical features improve the predictive power of machine-learning models that otherwise only rely on two-dimensional features like molecular fingerprints.
Additionally, we developed an internal benchmark set of experimental super-critical fluid chromatography retention times. For those methods, improvements of up to 10.6% are obtained when combining molecular fingerprints with physicochemical descriptors.
Shapley additive explanation values show furthermore that the physical nature of the applied features can be retained within the final machine-learning models.
We generally recommend the kallisto framework as a robust, low-cost, and physically motivated featurizer for upcoming state-of-the-art machine-learning studies.
