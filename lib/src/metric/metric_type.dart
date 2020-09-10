/// Metrics for measuring the quality of the prediction.
enum MetricType {
  /// Mean percentage absolute error (MAPE), a regression metric. The less the
  /// score produced by the metric, the better the prediction's quality is. Can
  /// lead to error if there are zero values among the original values. Normally,
  /// the metric produces scores within the range [0, 1], but extremely high
  /// predicted values (>> original values) can produce scores which are
  /// greater than 1.
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
}
