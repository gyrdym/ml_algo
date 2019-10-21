/// A type of kernel function.
///
/// Defines a way the weights of the nearest observations are being calculated
/// for K nearest neighbours algorithm (KNN).
///
/// Pure KNN algorithm does not consider how far is a particular observation
/// from the target one and it generally leads to imprecise prediction.
///
/// To avoid such a behaviour, one may use a kernel function, that accepts a
/// distance between an observation being evaluated and the target one and
/// returns a weight denoting, how much the evaluating observation will
/// contribute in prediction for the target one.
enum KernelType {
  /// A kernel, that calculates weights, using the following formula:
  ///
  ///
  /// ![K(x)=\left\{\begin{matrix}1/2, & |x|\leqslant \lambda \\ 0, & |x|> \lambda \end{matrix}\right](https://latex.codecogs.com/png.latex?K%28x%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D1/2%2C%20%26%20%7Cx%7C%5Cleqslant%20%5Clambda%20%5C%5C%200%2C%20%26%20%7Cx%7C%3E%20%5Clambda%20%5Cend%7Bmatrix%7D%5Cright)
  ///
  ///
  uniform,

  /// A kernel, that calculates weights, using the following formula:
  ///
  ///
  /// ![K(x)=\left\{\begin{matrix}3/4(1 - x^{2}), & |x|\leqslant \lambda \\ 0, & |x|> \lambda \end{matrix}\right](https://latex.codecogs.com/png.latex?K%28x%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D3/4%281%20-%20x%5E%7B2%7D%29%2C%20%26%20%7Cx%7C%5Cleqslant%20%5Clambda%20%5C%5C%200%2C%20%26%20%7Cx%7C%3E%20%5Clambda%20%5Cend%7Bmatrix%7D%5Cright)
  ///
  ///
  epanechnikov,

  /// A kernel, that calculates weights, using the following formula:
  ///
  ///
  /// ![K(x)=\left\{\begin{matrix} 0.25\pi cos(0.5\pi x) & |x|\leqslant \lambda \\ 0, & |x|>\lambda \end{matrix}\right](https://latex.codecogs.com/png.latex?K%28x%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200.25%5Cpi%20cos%280.5%5Cpi%20x%29%20%26%20%7Cx%7C%5Cleqslant%20%5Clambda%20%5C%5C%200%2C%20%26%20%7Cx%7C%3E%5Clambda%20%5Cend%7Bmatrix%7D%5Cright)
  ///
  ///
  cosine,

  /// A kernel, that calculates weights, using the following formula:
  ///
  ///
  /// ![K(x)=\frac{1 }{\sqrt{2\pi }} e^{-\frac{1}{2}x^{2}}](https://latex.codecogs.com/png.latex?K%28x%29%3D%5Cfrac%7B1%20%7D%7B%5Csqrt%7B2%5Cpi%20%7D%7D%20e%5E%7B-%5Cfrac%7B1%7D%7B2%7Dx%5E%7B2%7D%7D)
  ///
  ///
  gaussian,
}
