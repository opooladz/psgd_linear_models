# psgd_linear_models

Consider convergence of psgd vs sgd on liear models for classification and regression tasks


## Linear BinaryClassification 

PSGD performs the same as SGD for isotropic gaussian data
<img src="iso_sgd_psgd.png" width=90% height=90%>


PSGD significantly outperforms SGD for non-isotropic data
<img src="psgd_sgd_weibull.png" width=90% height=90%>

## Linear Regression

PSGD performs the same as SGD for isotropic gaussian data
<img src="mse_comparison_log.png" width=90% height=90%>

PSGD converges faster than SGD for non-isotropic data and to a much better solution
<img src="mse_comparison_log_weibull.png" width=90% height=90%>