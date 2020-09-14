/// Metrics for measuring the quality of the prediction.
enum MetricType {
  /// Mean percentage absolute error (MAPE), a regression metric. The formula
  /// is:
  ///
  ///
  /// ![{\mbox{Score}}={\frac{1}{n}}\sum_{{t=1}}^{n}\left|{\frac{Y_{t}-\widehat{Y}_{t}}{Y_{t}}}\right|](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%7B%5Cfrac%7B1%7D%7Bn%7D%7D%5Csum_%7B%7Bt%3D1%7D%7D%5E%7Bn%7D%5Cleft%7C%7B%5Cfrac%7BY_%7Bt%7D-%5Cwidehat%7BY%7D_%7Bt%7D%7D%7BY_%7Bt%7D%7D%7D%5Cright%7C)
  ///
  ///
  /// where Y - original value, Y with hat - predicted one
  ///
  /// The less the score produced by the metric, the better the prediction's
  /// quality is. Can lead to error if there are zero values among the original
  /// values. Normally, the metric produces scores within the range [0, 1]
  /// (both included), but extremely high predicted values (>> original values)
  /// can produce scores which are greater than 1.
  mape,

  /// Root mean squared error, a regression metric. The less the score produced
  /// by the metric, the better the prediction's quality is. The metric produces
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
