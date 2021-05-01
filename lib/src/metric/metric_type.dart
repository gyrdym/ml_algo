/// Metrics for measuring the quality of the prediction.
enum MetricType {
  /// Mean percentage absolute error (MAPE), a regression metric. The formula
  /// is:
  ///
  ///
  /// ![{\mbox{Score}}={\frac{1}{n}}\sum_{{t=1}}^{n}\left|{\frac{y_{t}-\widehat{y}_{t}}{y_{t}}}\right|](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%7B%5Cfrac%7B1%7D%7Bn%7D%7D%5Csum_%7B%7Bt%3D1%7D%7D%5E%7Bn%7D%5Cleft%7C%7B%5Cfrac%7By_%7Bt%7D-%5Cwidehat%7By%7D_%7Bt%7D%7D%7By_%7Bt%7D%7D%7D%5Cright%7C)
  ///
  ///
  /// where `y` - original value, `y` with hat - predicted one
  ///
  ///
  /// The less the score produced by the metric, the better the prediction's
  /// quality is. Can lead to error if there are zero values among the original
  /// values. Normally, the metric produces scores within the range [0, 1]
  /// (both included), but extremely high predicted values (>> original values)
  /// can produce scores which are greater than 1.
  mape,

  /// Root mean squared error (RMSE), a regression metric. The formula is
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

  /// Residual sum of squares (RSS), a regression metric. The formula is
  ///
  /// ![{\mbox{Score}}=\sum_{t=1}^{n}(y_{t} - \widehat{y}_{t})^{2}](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%5Csum_%7Bt%3D1%7D%5E%7Bn%7D%28y_%7Bt%7D%20-%20%5Cwidehat%7By%7D_%7Bt%7D%29%5E%7B2%7D)
  ///
  /// where `n` is a total amount of labels, `y` is an original value, `y` with
  /// hat - predicted one
  ///
  rss,

  /// A classification metric. The formula is
  ///
  ///
  /// ![{\mbox{Score}}=\frac{k}{n}](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%5Cfrac%7Bk%7D%7Bn%7D)
  ///
  ///
  /// where `k` is a number of correctly predicted labels, `n` - total amount
  /// of labels
  ///
  ///
  /// The greater the score produced by the metric, the better the prediction's
  /// quality is. The metric produces scores within the range [0, 1]
  accuracy,

  /// A classification metric. The formula for a single-class problem is
  ///
  ///
  /// ![{\mbox{Score}}=\frac{TP}{TP + FP}](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%5Cfrac%7BTP%7D%7BTP%20&plus;%20FP%7D)
  ///
  ///
  /// where `TP` is a number of correctly predicted positive labels (true positive),
  /// `FP` - a number of incorrectly predicted positive labels (false positive). In
  /// other words, `TP + FP` is a number of all the labels predicted to be positive
  ///
  /// The formula for a multi-class problem is
  ///
  ///
  /// ![{\mbox{Score}}= \frac{1}{n}\sum_{t=1}^{n}Score_{t}](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bt%3D1%7D%5E%7Bn%7DScore_%7Bt%7D)
  ///
  /// Where `Score 1..t` are scores for each class from 1 to t
  ///
  ///
  /// The greater the score produced by the metric, the
  /// better the prediction's quality is. The metric produces scores within the
  /// range [0, 1]
  precision,

  /// A classification metric. The formula for a single-class problem is
  ///
  ///
  /// ![{\mbox{Score}}=\frac{TP}{TP + FN}](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D)
  ///
  ///
  /// where `TP` is a number of correctly predicted positive labels (true positive),
  /// `FN` - a number of incorrectly predicted negative labels (false negative). In
  /// other words, `TP + FN` is a total amount of positive labels for a class in
  /// the given data
  ///
  /// The formula for a multi-class problem is
  ///
  ///
  /// ![{\mbox{Score}}= \frac{1}{n}\sum_{t=1}^{n}Score_{t}](https://latex.codecogs.com/gif.latex?%7B%5Cmbox%7BScore%7D%7D%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bt%3D1%7D%5E%7Bn%7DScore_%7Bt%7D)
  ///
  ///
  /// Where `Score 1..t` are scores for each class from 1 to t
  ///
  /// The greater the score produced by the metric, the
  /// better the prediction's quality is. The metric produces scores within the
  /// range [0, 1]
  recall,
}
