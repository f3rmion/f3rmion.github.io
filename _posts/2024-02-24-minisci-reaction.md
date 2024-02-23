## How to model chemical reactivity with machine learning

## Model selection _via_ lazy predict

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
