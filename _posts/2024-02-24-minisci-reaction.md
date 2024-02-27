## Model chemical reactivity with machine learning

In this blog post, I would like to show you how to model chemical reactivity _via_ machine learning by exemplifying this for the prediction of regioselectivity in radical C-H functionalization of heteroaromatic systems.

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

Input some fancy reaction mechanism here

Remember how I said earlier that the outcome of the Minisci reaction is heavily influenced by the nature of the underlying radical and the heteroaromatic system?
This simply means that the nature of the reactants is somehow responsible for the outcome, which makes sense on an intuitive level.
And indeed, the scientific literature has shown that predictive models based on some chemical descriptors of the reactants is indeed very valuable if combined with, e.g.,regression techniques.

We call such regression models _quantitative structure-activity relationship_ models since we relate a set of variables (_X_) to some response variable (_Y_).
A very simple example of regression is given by linear regression where we try to predict the value of one variable _Y_ by another _X_.
If we include more input variables we can use multi-linear regression to couple multiple independent variables to predict another.
Hence, we need some function _F_ that enables us to create a mapping between our inputs _X_ and the output _Y_.

In the case of linear regression, this function is quite easy to find.
However, if we want to apply a non-linear mapping function we need some way to automatically find this function.
Luckily, there exist quite awesome Python libraries that simplify the process of finding such mappings.
And in the coming sections, we will use them to obtain the mappings that will enable us to predict outcomes by a set of descriptors based on the reactants.
However, let's dive first into what we actually want to predict, which brings us to the next important topic - data mining.

### Mining the literature for appropiate data

If we start working on a project it is always very useful to check out the literature to see who has already published something similar.
For the prediction of regioselectivity for the Minisci reaction, [Zhang et al.](https://onlinelibrary.wiley.com/doi/10.1002/anie.202000959) have published a great work in 2020 in Angewante Chemie.
In this work, the authors used a machine learning model to predict the [transition state](https://en.wikipedia.org/wiki/Transition_state_theory) barrier from computed properties of the isolated reactants, which is supposed to enable the rapid regioselectivity prediction for radical C-H bond functionalisation of herterocycles.
In this work, they used machine learning to find the mapping between physical organic features (_X_) and some theoretically derived barriers (_Y_).
They used [density functional theory](https://en.wikipedia.org/wiki/Density_functional_theory) to compute activation barriers and conveniently provided their data inside a [GitHub](https://github.com/Masker-Li/ChemSelML/blob/master/DataSet/Canonicalized_SMILES_Reactions_input_data.csv) repository.

This is by far the ideal case to start a project, but keep in mind that most of the time mining the literature is **hard work** and is followed by multiple rounds of [cleaning the data](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/) and [outlier detection](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/).
For the sake of brevity, we won't cover those topics in this blog post.
However, make always sure that the underlying data is meaningful because machine-learning models will follow along the well-known [GIGO](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out) principle.

### Machine-learning mapping functions

Alright, we are in this blog post already quite some time.
Let's dig into the essence of what we actually wanted to do.
Train some machine-learning mapping function to enable us to make some predictions about regioselectiovity.
But how should we start?
Which model should we test?
This is a great question ans often times we simply need to test quite a few different ones to get a feeling for which of them is a reasonable choice for the underlying data.
A rule of thumb for myself is to start with a [random forest](https://en.wikipedia.org/wiki/Random_forest) model first and from there work my way through more complex models.

Recently, a great Python package called [`lazypredict`](https://lazypredict.readthedocs.io/en/latest/) was introduced to that simplifies the screening of models.

### GFN-xTB lazypredict

| Model                         | Adjusted R-Squared | R-Squared   | RMSE    | Time Taken |
| ----------------------------- | ------------------ | ----------- | ------- | ---------- |
| XGBRegressor                  | 0.97               | 0.98        | 0.72    | 0.74       |
| HistGradientBoostingRegressor | 0.97               | 0.97        | 0.75    | 0.44       |
| LGBMRegressor                 | 0.97               | 0.97        | 0.75    | 0.28       |
| ExtraTreesRegressor           | 0.95               | 0.96        | 0.99    | 3.13       |
| RandomForestRegressor         | 0.95               | 0.95        | 1.05    | 8.95       |
| BaggingRegressor              | 0.94               | 0.94        | 1.14    | 0.72       |
| MLPRegressor                  | 0.92               | 0.93        | 1.25    | 4.80       |
| DecisionTreeRegressor         | 0.91               | 0.91        | 1.38    | 0.13       |
| GradientBoostingRegressor     | 0.90               | 0.90        | 1.45    | 1.80       |
| ExtraTreeRegressor            | 0.86               | 0.87        | 1.68    | 0.06       |
| SVR                           | 0.86               | 0.87        | 1.70    | 3.03       |
| NuSVR                         | 0.86               | 0.87        | 1.72    | 2.56       |
| KNeighborsRegressor           | 0.85               | 0.86        | 1.74    | 0.09       |
| AdaBoostRegressor             | 0.75               | 0.76        | 2.30    | 0.70       |
| TransformedTargetRegressor    | 0.72               | 0.73        | 2.43    | 0.03       |
| LinearRegression              | 0.72               | 0.73        | 2.43    | 0.02       |
| BayesianRidge                 | 0.72               | 0.73        | 2.43    | 0.06       |
| RidgeCV                       | 0.71               | 0.72        | 2.46    | 0.15       |
| Ridge                         | 0.69               | 0.71        | 2.54    | 0.02       |
| HuberRegressor                | 0.68               | 0.70        | 2.56    | 0.19       |
| LinearSVR                     | 0.68               | 0.70        | 2.58    | 0.18       |
| LassoCV                       | 0.68               | 0.70        | 2.59    | 0.58       |
| ElasticNetCV                  | 0.66               | 0.68        | 2.66    | 0.31       |
| SGDRegressor                  | 0.66               | 0.68        | 2.67    | 0.23       |
| RANSACRegressor               | 0.65               | 0.67        | 2.71    | 0.35       |
| PoissonRegressor              | 0.58               | 0.61        | 2.94    | 0.06       |
| OrthogonalMatchingPursuitCV   | 0.47               | 0.50        | 3.33    | 0.21       |
| LassoLarsIC                   | 0.46               | 0.49        | 3.35    | 0.04       |
| GammaRegressor                | 0.40               | 0.43        | 3.53    | 0.05       |
| TweedieRegressor              | 0.40               | 0.43        | 3.54    | 0.04       |
| LassoLarsCV                   | 0.34               | 0.38        | 3.70    | 0.07       |
| OrthogonalMatchingPursuit     | 0.33               | 0.36        | 3.74    | 0.03       |
| LarsCV                        | 0.31               | 0.34        | 3.80    | 0.12       |
| ElasticNet                    | 0.28               | 0.32        | 3.88    | 0.02       |
| PassiveAggressiveRegressor    | 0.18               | 0.23        | 4.12    | 0.09       |
| Lasso                         | 0.17               | 0.21        | 4.16    | 0.02       |
| LassoLars                     | 0.17               | 0.21        | 4.16    | 0.03       |
| QuantileRegressor             | -0.06              | -0.00       | 4.69    | 1.73       |
| DummyRegressor                | -0.06              | -0.00       | 4.69    | 0.02       |
| GaussianProcessRegressor      | -0.17              | -0.11       | 4.93    | 4.04       |
| KernelRidge                   | -10.42             | -9.81       | 15.41   | 1.55       |
| Lars                          | -1645379.70        | -1557037.11 | 5848.83 | 0.04       |

```

```
