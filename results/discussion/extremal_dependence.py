# Work in progress!

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import ExponentiatedWeibullDistribution, MultivariateDistribution

u = 0.9999
#u = 1 - 1 / (50 * 365.25 * 24)

# Create distribution.
U10 = ExponentiatedWeibullDistribution(shape=2.42, scale=10.0, shape2=0.761)
hs_shape = FunctionParam('logistics4', 0.582, 1.90, 0.248, 8.49)
hs_scale = FunctionParam('alpha3', 0.394, 0.0178, 1.88,
                            C1=0.582, C2=1.90, C3=0.248, C4=8.49)
hs_shape2 = ConstantParam(5)
Hs = ExponentiatedWeibullDistribution(shape=hs_shape, scale=hs_scale, shape2=hs_shape2)
distributions = [U10, Hs]
dependencies = [(None, None, None, None) , (0, None, 0, None) ]
mul_dist = MultivariateDistribution(distributions, dependencies)

ur = U10.i_cdf(u)
#hr = mul_dist.marginal_icdf(u, dim=1)
hr = 8.5
print('u_r = ' + str(ur) + ', h_r = ' + str(hr))
jointAB = 1 - U10.cdf(ur) - mul_dist.marginal_cdf(hr, dim=1) + mul_dist.cdf([ur, hr], lower_integration_limit=(0, 0))
print('jointAB = ' + str(jointAB))
lam = jointAB / (1 - u)
print('lambda = ' + str(lam))
