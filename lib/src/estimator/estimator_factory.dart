import 'estimator.dart';
import 'estimator_type.dart';
import 'rmse.dart';
import 'mape.dart';

class EstimatorFactory {
  static Estimator create(EstimatorType type) {
    switch (type) {
      case EstimatorType.RMSE:
        return new RMSEEstimator();

      case EstimatorType.MAPE:
        return new MAPEEstimator();

      default:
        throw new UnsupportedError('Estimator type $type is not supported!');
    }
  }
}
