## Model chemical reactivity with machine learning

In this blog post, I would like to present how one can model chemical reactivity _via_ machine learning by exemplifying this for the prediction of regioselectivity determination in radical C-H functionalization of heteroaromatic systems.

---

### Implementation and training set

- [GitHub repository](https://github.com/C-H-activation/minisci/tree/main)
- [Machine-learning training set]()

### Mentioned dependencies

- `lazypredict`: machine-learning screening library (Python)
- `GFN-xTB`: semi-empirical tight-binding model (Fortran)
- `Morfeus`: molecular featuriser for machine-learning (Python)
- `SHAP` analysis: game theoretical analysis to check feature importance

---

#### Prepare the stage

The basic idea is to create a machine-learning model that enables us to predict where a radical will create a new bond within an aromatic system that contains a heteroatom (atom different than carbon).
Such a model is of great interest, since it could allow us to gain intuition and more insights into so far unknown territories and could speed up predictions for this kind of reaction.

The reaction that we model is called the Minisci reaction, which is a named reaction within organic chemistry.
This reaction enables us to perform a radical substitution to an aromatic compound.
The reaction was published in 1971 by [F. Minisci](https://doi.org/10.1016%2Fs0040-4020%2801%2997768-3).
The Minisci reaction often produces a wild mixture of [regioisomers](https://en.wikipedia.org/wiki/Regioisomer), however, modern versions of this reaction allow a wide range of alkygroups to be introduced.

One complication relies in the fact that the outcome heavily depends both on the radical source and the heteroaromatic system.
However, since this reaction allows for alkylation of electron deficient systems - which is not possible by [Friedel-Crafts chemistry](https://en.wikipedia.org/wiki/Friedel%E2%80%93Crafts_reaction) - it is often used for alkylation chemistry.

Imagine how a chemist could use their intuition to predict the outcome of a Minisci reaction.
Next to deep knowledge in the mechanistics of this reaction type they probably also need a decent amount of time to predict the outcome in a proper way.
What will we do if we want to screen 100 reactions? What if we want to screen 10,000? The poor chemist could either invest a lot of time into this process or we try to automatise it somehow.
Let's try to get a deeper understanding of this reaction type first.

#### The Minisci reaction in more detail

A free radical is formed by the radical starter within the reaction.
Often times one uses an acid motif (`-COOH`) in combination with a silver salt to initiate a so-called oxidative decarboxylation.
This is quite a fancy name and it means that we abstract a Hydrogen atom by the silver, followed by a release of CO2 as gas to form some kind of radical.
This radical then reacts with the heteroaromatic compound and the final product is formed by re-aromatization.
The following simplified reaction sketch is showing the mechanism in more detail.

![Minisci mechanism](https://raw.githubusercontent.com/f3rmion/f3rmion.github.io/refs/heads/main/_posts/assets/2024-02-24/minisci-mechanism.png)

Remember how I said earlier that the outcome of the Minisci reaction is heavily influenced by the nature of the underlying radical and the heteroaromatic system?
This simply means that the nature of the reactants is somehow responsible for the outcome, which makes sense on an intuitive level.
And indeed, the scientific literature has shown that predictive models based on some chemical descriptors of the reactants is indeed very valuable if combined with, e.g.,regression techniques.

We call such regression models _quantitative structure-activity relationship_ models since we relate a set of variables (_X_) to some response variable (_Y_).
A very simple example of regression is given by linear regression where we try to predict the value of one variable by another.
If we include more input variables we can use multi-linear regression to couple multiple independent variables to predict another.
Hence, we need some function that enables us to create a mapping between our inputs and the output.

In the case of linear regression, this function is quite easy to find.
However, if we want to apply a non-linear mapping function we need some way to automatically find this function.
Luckily, there exist awesome libraries that simplify the process of finding such mappings.
And in the coming sections, we apply them to obtain the mappings that will enable us to predict outcomes by a set of descriptors based on the reactants.
However, let's dive first into what we actually want to predict, which brings us to the next important topic - data mining.

### Mining the literature for appropiate data

If we start working on a project it is always very useful to check out the literature to see who has already published something similar.
For the prediction of regioselectivity for the Minisci reaction, [Zhang et al.](https://onlinelibrary.wiley.com/doi/10.1002/anie.202000959) have published a great work in 2020 in Angewante Chemie.
In this work, the authors used a machine learning model to predict the [transition state](https://en.wikipedia.org/wiki/Transition_state_theory) barrier from computed properties of the isolated reactants.
In this work, they used machine learning to find the mapping between physical organic features (_X_) and some theoretically derived barriers (_Y_).
You can check their [supporting information](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002/anie.202000959&file=anie202000959-sup-0001-misc_information.pdf) to get a feeling for what kind of features they selected (in total 50 different features).
They used [density functional theory](https://en.wikipedia.org/wiki/Density_functional_theory) (DFT) to compute activation barriers and conveniently provided their data inside a [GitHub](https://github.com/Masker-Li/ChemSelML/blob/master/DataSet/Canonicalized_SMILES_Reactions_input_data.csv) repository.

This is by far the ideal case to start a project, but keep in mind that most of the time mining the literature is **hard work** and is followed by multiple rounds of [cleaning the data](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/) and [outlier detection](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/).
For the sake of brevity, we won't cover those topics in this blog post.
However, make always sure that the underlying data is meaningful because machine-learning models will follow along the well-known [GIGO](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out) principle.

### Machine-learning mapping functions

Let's dig into the essence of what we actually wanted to do.
Train some machine-learning mapping function to enable us to make some predictions about regioselectiovity.
But how should we start?
Which model should we test?
This is a great question and often times we simply need to test quite a few different ones to get a feeling for which of them is a reasonable choice for the underlying data.
A rule of thumb for myself is to start with a [random forest](https://en.wikipedia.org/wiki/Random_forest) model first and from there work my way through more complex models.

Recently, a great Python package called [`lazypredict`](https://lazypredict.readthedocs.io/en/latest/) was introduced to me that simplifies the screening of models and we use it in this blog post to exemplify model performances.
Before we jump into model fitting, we need to clarify which features we want to use to describe the reactants.
In their paper, Zhang et al. used several different feature sets, but the best results were achieved by a random forest model combined with physical organic features (94.2% site selectivity).
The authors used [Gaussian](https://gaussian.com/) to optimise chemical structures and to calculate features.
Since this software is commercial ($$$), we choose the free and open-source project [GFN-xTB](https://github.com/grimme-lab/xtb) as a feature backend, which is a semi-empirical [tight binding](https://en.wikipedia.org/wiki/Tight_binding) model.

This software is written as a command-line interface (CLI) and we could wrap this CLI with output readers and extract the features ourselves.
However, here we make use of another project that already created a great wrapping Python package, called [Morfeus](https://digital-chemistry-laboratory.github.io/morfeus/).
This package can be imported as a Python module to simplify the interaction with xTB.
In the next section, we dive deeper into what features we extract from both reactants.

### Designing a feature vector

Next, we want to understand how to design a feature vector that is used to train a mapping to a certain reference outcome.
Morfeus gives us the `XTB` class that we can use to create an instance that gives us class methods to calculate features with.
Let's take a water molecule as example and featurise (describe) it.
We represent water in three-dimensions (`xmol` format) and the coordinates that we apply are as follows

```markdown
$ cat water.xyz
3

o 0.00000000000000 -0.06365718653467 -0.00000000010001
h 0.77223547167307 0.50524694437309 0.00000000070010
h -0.77223547197311 0.50524694397304 0.00000000030004
```

We use this molecule and calculate a few physical organic features using xTB.
Note that you need to have [xTB installed](https://xtb-docs.readthedocs.io/en/latest/setup.html) on your system to follow along - you can easily install xTB _via_ Conda.
Now we build a small script that reads the water coordinates and we calculate features like atomic [partial charges](https://en.wikipedia.org/wiki/Partial_charge), [Fukui indices](https://en.wikipedia.org/wiki/Fukui_function), and electonical properties like the highest-occupied molecular orbital (HOMO) or the lowest-unoccupied molecular orbital (LUMO) as obtained from [molecular-orbital theory](https://en.wikipedia.org/wiki/Molecular_orbital_theory).

```python
from morfeus import read_xyz
from morfeus import XTB

# get elements and coordinates from xmol file
water_elements, water_coordinates = read_xyz("water.xyz")

# build XTB instance for a neutral water molecule
molecular_charge = 0
water_xtb = XTB(
    elements=water_elements,
    coordinates=water_coordinates,
    charge=molecular_charge,
)

# partial charges: {1: -0.5647323737329832, 2: 0.28236618686659615, 3: 0.28236618686638876}
water_charges = water_xtb.get_charges()

# electrophilicity fukui: {1: 0.22775662534546415, 2: 0.38612168731709295, 3: 0.3861216873374471}
water_electrophilicity_fukui = water_xtb.get_fukui("electrophilicity")

# nucleophilicity fukui: {1: 0.5973889157529119, 2: 0.2013055421228951, 3: 0.20130554212419272}
water_nucleophilicity_fukui = water_xtb.get_fukui("nucleophilicity")

# ionisation potential: 13.44750237224477
water_ip = water_xtb.get_ip(corrected=True)

# electron affinity: -6.6728543165601835
water_ea = water_xtb.get_ea()

# chemical potential (mu): -10.060178344402477
water_mu = (water_ea - water_ip) / 2.0

# chemical hardness (eta): 20.120356688804954
water_eta = water_ip - water_ea

# homo energy: -0.4463812902311759
water_homo = water_xtb.get_homo()

# lumo energy: 0.08246796400242422
water_lumo = water_xtb.get_lumo()
```

See the feature selection section further below to get more insights about how to choose features by exploiting a game theoretical approach that explains the outcome of a machine-learning model based on each feature's contribution to the model.
In the next section, we want to talk about model selection to check which machine-learning model is performing reasonably for the problem of interest.

### Machine-learning model selection

We take the [final training set]() to test `lazypredict`'s ability to quickly screen different machine-learning models.
We use the following code snippet for the screening:

```python
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from lazypredict.Supervised import LazyRegressor

df = pd.read_csv("X_physchem.csv")

# length of the feature vector
feature_length = 32

# extract training data: X
# extract reference values (column "dg_ts"): Y
X_df = df.iloc[0: , 1:feature_length + 1]
y_df = df["dg_ts"]

# randomly shuffle data
X, y = shuffle(X_df, y_df, random_state=13)
X = X.astype(np.float32)

# train machine-learning models
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# perform regression
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
```

The table below lists statistical measures for 41 different machine-learning models:

| Model                         | R_a         | R           | RMSE    | t    |
| ----------------------------- | ----------- | ----------- | ------- | ---- |
| XGBRegressor                  | 0.97        | 0.98        | 0.72    | 0.74 |
| HistGradientBoostingRegressor | 0.97        | 0.97        | 0.75    | 0.44 |
| LGBMRegressor                 | 0.97        | 0.97        | 0.75    | 0.28 |
| ExtraTreesRegressor           | 0.95        | 0.96        | 0.99    | 3.13 |
| RandomForestRegressor         | 0.95        | 0.95        | 1.05    | 8.95 |
| BaggingRegressor              | 0.94        | 0.94        | 1.14    | 0.72 |
| MLPRegressor                  | 0.92        | 0.93        | 1.25    | 4.80 |
| DecisionTreeRegressor         | 0.91        | 0.91        | 1.38    | 0.13 |
| GradientBoostingRegressor     | 0.90        | 0.90        | 1.45    | 1.80 |
| ExtraTreeRegressor            | 0.86        | 0.87        | 1.68    | 0.06 |
| SVR                           | 0.86        | 0.87        | 1.70    | 3.03 |
| NuSVR                         | 0.86        | 0.87        | 1.72    | 2.56 |
| KNeighborsRegressor           | 0.85        | 0.86        | 1.74    | 0.09 |
| AdaBoostRegressor             | 0.75        | 0.76        | 2.30    | 0.70 |
| TransformedTargetRegressor    | 0.72        | 0.73        | 2.43    | 0.03 |
| LinearRegression              | 0.72        | 0.73        | 2.43    | 0.02 |
| BayesianRidge                 | 0.72        | 0.73        | 2.43    | 0.06 |
| RidgeCV                       | 0.71        | 0.72        | 2.46    | 0.15 |
| Ridge                         | 0.69        | 0.71        | 2.54    | 0.02 |
| HuberRegressor                | 0.68        | 0.70        | 2.56    | 0.19 |
| LinearSVR                     | 0.68        | 0.70        | 2.58    | 0.18 |
| LassoCV                       | 0.68        | 0.70        | 2.59    | 0.58 |
| ElasticNetCV                  | 0.66        | 0.68        | 2.66    | 0.31 |
| SGDRegressor                  | 0.66        | 0.68        | 2.67    | 0.23 |
| RANSACRegressor               | 0.65        | 0.67        | 2.71    | 0.35 |
| PoissonRegressor              | 0.58        | 0.61        | 2.94    | 0.06 |
| OrthogonalMatchingPursuitCV   | 0.47        | 0.50        | 3.33    | 0.21 |
| LassoLarsIC                   | 0.46        | 0.49        | 3.35    | 0.04 |
| GammaRegressor                | 0.40        | 0.43        | 3.53    | 0.05 |
| TweedieRegressor              | 0.40        | 0.43        | 3.54    | 0.04 |
| LassoLarsCV                   | 0.34        | 0.38        | 3.70    | 0.07 |
| OrthogonalMatchingPursuit     | 0.33        | 0.36        | 3.74    | 0.03 |
| LarsCV                        | 0.31        | 0.34        | 3.80    | 0.12 |
| ElasticNet                    | 0.28        | 0.32        | 3.88    | 0.02 |
| PassiveAggressiveRegressor    | 0.18        | 0.23        | 4.12    | 0.09 |
| Lasso                         | 0.17        | 0.21        | 4.16    | 0.02 |
| LassoLars                     | 0.17        | 0.21        | 4.16    | 0.03 |
| QuantileRegressor             | -0.06       | -0.00       | 4.69    | 1.73 |
| DummyRegressor                | -0.06       | -0.00       | 4.69    | 0.02 |
| GaussianProcessRegressor      | -0.17       | -0.11       | 4.93    | 4.04 |
| KernelRidge                   | -10.42      | -9.81       | 15.41   | 1.55 |
| Lars                          | -1645379.70 | -1557037.11 | 5848.83 | 0.04 |

The [R measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) is the coefficient of determination, while the adjusted R measure ($R_a$) is a modified version that has been slightly tweaked to determine how reliable the correlation is and how much it is determined by the addition of independent variables:

$$
R_a = 1 - (1 - R) (n - 1)/(n - p - 1)
$$

In the equation above, $p$ is the total number of variables in the model and $n$ is the sample size.
Furthermore, we list the root-mean squares error (RMSE) as well as the training time needed ($t$) in seconds.
The reference values are calculated barriers of activation in kcal/mol.

We confirm that a trained random forest model performs similarly well to the model that the authors presented in their paper even though we substituted the orders of magnitude more computationally intense DFT calculations by the semi-empirical xTB ones.
Furthermore, we see that gradient boosted regression models outperform a random forest model.

### Feature selection

[Shapley additive explanations](https://shap.readthedocs.io/en/latest/) analysis is used for testing the feature importance.
This analysis is based on a game theoretical approach that explains the outputs of a machine-learning model based on the feeded features (see their [papers](https://shap.readthedocs.io/en/latest/) for more information).

We use the `XGBRegressor` machine-learning model as example since it performed the best in our analysis earlier.
Note that feature selection and machine-learning accuracy determination are steps that need to be repeated.
In general, one wants to find a good feature coverage to express most of the example in the training set with confidence.
There is kind of a cycle of optimization that one enters until the accuracy converges with a set of features that is then used to create the final model.

The code below trains the `XGBRegressor` model and outputs SHAP values:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost

# create Pandas data frame
df = pd.read_csv('X_physchem.csv')
length = 32
X_df = df.iloc[: , 1:length+1]

# names of feature columns (in order wrt the csv)
# c: compound (p: cation, m: anion)
# r: radical (p: cation, m: anion)
feature_names = ['q_c', 'q_cp', 'q_cm', 'c_ip', 'c_ea', 'c_homo', 'c_lumo', 'c_e', 'c_n', 'c_le', 'c_ln', 'c_cp', 'c_ch', 'c_fbv3', 'c_fbv4', 'c_fbv5', 'q_r', 'q_rp', 'q_rn', 'r_ip', 'r_ea', 'r_homo', 'r_lumo', 'r_e', 'r_n', 'r_le', 'r_ln', 'r_cp', 'r_ch', 'r_fbv3', 'r_fbv4', 'r_fbv5']

# name of reference activation energy in kcal/mol
y_df = df['dg_ts']

X = np.array(X_df)
y = np.array(y_df)

# define machine-learning model and train it
xgb = xgboost.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=3)
xgb.fit(X, y)

# SHAP analysis
shap_explainer = shap.TreeExplainer(xgb)
shap_values = shap_explainer.shap_values(X)
plt = plt.figure()
shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="layered_violin", color="coolwarm")
plt.savefig("xgb_shap_analysis.png", bbox_inches='tight', dpi=300)
```

The generated figure is shown below, where each feature is represented by violin plots and the feature value is represented by gradient colors, where red represents a high feature value and blue a low one.

<p align="center">
<img src="https://raw.githubusercontent.com/f3rmion/f3rmion.github.io/refs/heads/main/_posts/assets/2024-02-24/xgb_shap.png" alt="SHAP analysis" width="500"/>
</p>

The complete feature composition is shared in the [GitHub resource](https://github.com/C-H-activation/minisci/blob/main/minisci/helper/features.py).
