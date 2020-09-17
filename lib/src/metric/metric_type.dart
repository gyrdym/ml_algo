/// Metrics for measuring the quality of the prediction.
enum MetricType {
  /// Mean percentage absolute error (MAPE), a regression metric. The formula
  /// is:
  ///
  ///
  /// ![{\mbox{Score}}={\frac{1}{n}}\sum_{{t=1}}^{n}\left|{\frac{y_{t}-\widehat{y}_{t}}{y_{t}}}\right|](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%7B%5Cfrac%7B1%7D%7Bn%7D%7D%5Csum_%7B%7Bt%3D1%7D%7D%5E%7Bn%7D%5Cleft%7C%7B%5Cfrac%7By_%7Bt%7D-%5Cwidehat%7By%7D_%7Bt%7D%7D%7By_%7Bt%7D%7D%7D%5Cright%7C)
  ///
  ///
  /// where y - original value, y with hat - predicted one
  /// 
  ///
  /// The less the score produced by the metric, the better the prediction's
  /// quality is. Can lead to error if there are zero values among the original
  /// values. Normally, the metric produces scores within the range [0, 1]
  /// (both included), but extremely high predicted values (>> original values)
  /// can produce scores which are greater than 1.
  mape,

  /// Root mean squared error (RMSE), a regression metric. The formula is:
  ///
  ///
  /// ![{\mbox{Score}}=\sqrt{\frac{1}{n}\sum_{{t=1}}^{n}({\widehat{y}_{t} - y_{t}})^2}](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7B%7Bt%3D1%7D%7D%5E%7Bn%7D%28%7B%5Cwidehat%7By%7D_%7Bt%7D%20-%20y_%7Bt%7D%7D%29%5E2%7D)
  ///
  ///
  ///  where `y` is an original value, `y` with hat - predicted one
  ///
  ///
  /// The less the score
  /// produced by the metric, the better the prediction's quality is. The metric produces
  /// scores within the range [0, +Infinity]
  rmse,

  /// A classification metric. The greater the score produced by the metric, the
  /// better the prediction's quality is. The metric produces scores within the
  /// range [0, 1]
  accuracy,

  /// A classification metric. The greater the score produced by the metric, the
  /// better the prediction's quality is. The metric produces scores within the
  /// range [0, 1]
  precision,

  /// A classification metric. The greater the score produced by the metric, the
  /// better the prediction's quality is. The metric produces scores within the
  /// range [0, 1]
  recall,
}
