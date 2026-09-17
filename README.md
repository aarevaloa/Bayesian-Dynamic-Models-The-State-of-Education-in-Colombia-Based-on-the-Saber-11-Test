# Bayesian Dynamic Models: The State of Education in Colombia Based on the Saber 11 Test

**Authors:** Andrés Felipe Arévalo Arévalo - Juan Camilo Sosa Martínez·

---

## Abstract

The assessment of educational quality through standardized testing has become a central input for the design, monitoring, and adjustment of public education policies. In Colombia, data from the Saber 11 examination provide a rich and periodic source of information on students' academic performance and on their individual, municipal, and departmental contexts. However, most existing empirical studies rely on essentially static approaches that do not explicitly model the temporal evolution of covariate effects nor the hierarchical heterogeneity across levels. This limitation restricts the understanding of the dynamics of educational gaps and of the relative importance of explanatory factors over time.

This study proposes a hierarchical and dynamic Bayesian framework with LASSO regularization for the analysis of data with a nested temporal and territorial structure. Methodologically, four models are formulated by crossing two factors: the specification of the mean, static or dynamic, and the observation distribution, Normal or Student-t, the latter aimed at capturing heavy tails and outliers. All four share a three-level hierarchical structure, with students nested within municipalities within departments, covariates measured at those same levels, and Bayesian LASSO regularization with a separate penalization parameter for each level of measurement. In the dynamic specifications, regression coefficients are treated as latent states that evolve over time according to state equations, allowing for an explicit modeling of the dynamics of covariate effects, while shrinkage is applied to the innovations rather than to the coefficients, so that the regularized hypothesis is not that effects are null but that they remain stable across periods. The proposed framework is implemented using Markov chain Monte Carlo (MCMC) algorithms that combine forward filtering backward sampling (FFBS) for the state trajectory, retrospective beta-gamma sampling for the evolution of the observational variances, and a Gibbs sampler for the hierarchical and regularization parameters.

The four models are fitted to nearly five hundred thousand students distributed across 1,107 municipalities and 33 departments over sixteen periods, and are compared in terms of predictive performance through the expected log pointwise predictive density (elpd), complemented by the WAIC and DIC criteria. The results show that the dynamic factor dominates the robustness factor by a wide margin, and that the two factors interact, in the sense that under a static specification the Student-t likelihood does not improve predictive performance, whereas under a dynamic specification it improves it appreciably. The dynamic model with Student-t likelihood emerges as the formal predictive optimum.

<img width="1420" height="1528" alt="fig06_nivel_muni" src="https://github.com/user-attachments/assets/975529bc-14e8-4c97-aa9a-fe683e1775df" />
<img width="2254" height="1560" alt="fig01_encuadre" src="https://github.com/user-attachments/assets/1b89366d-ec26-4ba4-9e27-10ad4a4f02cf" />

---

## Models

| Model | Mean specification | Observation distribution | Main algorithmic components |
|:--|:--|:--|:--|
| M1 | Static | Normal | Gibbs sampler + Bayesian LASSO by level |
| M2 | Static | Student-t | Gibbs sampler + Normal-gamma scale mixture |
| M3 | Dynamic | Normal | FFBS + dynamic LASSO on the innovations |
| M4 | Dynamic | Student-t | FFBS + beta-gamma variance evolution + Metropolis-Hastings step |

---

## Contents

The repository is organized in two top-level folders, `Codes` and `Results`, both of which follow the same internal structure by model.

### `Codes`

This folder contains three subfolders:

* **`Data_Manipulation`** — code used to build and prepare the databases: the ICFES Saber 11 records, the municipal and departmental covariates, and the nested student / municipality / department structure observed over the sixteen periods. The prepared files constitute the starting point of the analysis.
* **`Models`** — code of the samplers of the four models, together with their comparison and the analysis of the state of education in Colombia. It contains one subfolder per model (`Model 1`, `Model 2`, `Model 3`, `Model 4`), a `Comparison` subfolder with the uniform evaluation of the four fitted models, and a `Colombian_Education` subfolder with the substantive analysis based on the selected model.
* **`Simulations`** — code used for the simulation exercises with which the samplers and the recovery of the parameters were assessed.

Every module has a **runner**, that is, a single script that assembles the complete sampler, analysis or comparison, so that the settings of the run — dataset, burn-in period, thinning step and output paths — are handled in one place and the code can be executed easily.

| Stage | Runner |
|:--|:--|
| Model 1 | `16_Correr_Muestreador_M1.R` |
| Model 2 | `10_gibbs_samper_M2.R` |
| Model 3 | `8_gibbs_M3.R` |
| Model 4 | `8_Muestreador_GIBBS_v4_M4.R` |
| Comparison of the four models | `ejecutar_comparaciones_v2.ipynb` |
| Analysis | `ejecutar_M4.ipynb` |

The following is the workflow to reproduce the results obtained in the analysis.

* **Preparation**

  * Run the scripts in `Codes/Data_Manipulation` to obtain the prepared dataset with the covariates measured at the student, municipal and departmental levels.

* **Samplers**

  * Run `16_Correr_Muestreador_M1.R` to obtain the MCMC output of the static Normal model.
  * Run `10_gibbs_samper_M2.R` to obtain the MCMC output of the static Student-t model.
  * Run `8_gibbs_M3.R` to obtain the MCMC output of the dynamic Normal model.
  * Run `8_Muestreador_GIBBS_v4_M4.R` to obtain the MCMC output of the dynamic Student-t model.

* **Comparison and analysis**

  * Run `ejecutar_comparaciones_v2.ipynb` to obtain the comparison of the four models under a single evaluation code.
  * Run `ejecutar_M4.ipynb` to obtain the diagnostics, the figures and the analysis of the state of education in Colombia.

* **Simulations**

  * Run the scripts in `Codes/Simulations` to reproduce the simulation exercises.

### `Results`

This folder contains the figures that summarize part of the results and some of the chains of the parameters evaluated in each model, organized with the same structure by model used in `Codes/Models`. Because of storage limitations it was not possible to upload the complete set of outputs; should you need access to all of the results, please contact the owners of the repository.

---

## Data

The analysis is based on the ICFES Saber 11 records, publicly available through the ICFES open data portal, complemented with municipal and departmental indicators and with the DANE cartography used for the territorial maps.

## Requirements

* **R** _[versión]_ for the samplers and the preparation of the data.
* **Python** _[versión]_ for the diagnostics, the comparison of the models and the figures, executed through Jupyter notebooks.

## Citation

```bibtex
@mastersthesis{[clave],
  author  = {[Apellido, Nombre]},
  title   = {Bayesian Dynamic Models: The State of Education in Colombia Based on the Saber 11 Test},
  school  = {Universidad Nacional de Colombia},
  year    = {[año]},
  type    = {Tesis de Maestr\'ia en Ciencias --- Estad\'istica}
}
```

## Contact

_[correo de contacto]_
