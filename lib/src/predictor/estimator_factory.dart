part of linear_regressor;

class _EstimatorFactory {
  static Estimator create(EstimatorType type) {
    switch (type) {
      case EstimatorType.RMSE:
        return new RMSEEstimator();

      default:
        throw new UnsupportedError('Estimator type $type is not supported!');
    }
  }
}
