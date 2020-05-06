import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';

/// Regularization types
///
/// Machine learning linear models are prone to overfitting, or, in other words,
/// they may lose generalization ability in their learning process: overfitted
/// models show quite high quality on training data, but in the same time they
/// perform very badly on previously unseen data.
///
/// The main reason of that is uncontrolled growth of linear model coefficients.
/// To avoid this, it is needed to measure a magnitude of coefficients vector
/// and consider it during the model's learning.
enum RegularizationType {
  ///uses Manhattan norm of a vector to calculate magnitude of learned
  ///coefficients. Applicable for [LinearOptimizerType.coordinate]
  // ignore: constant_identifier_names
  L1,

  ///uses Euclidean norm of a vector to calculate magnitude of learned
  ///coefficients. Applicable for [LinearOptimizerType.gradient]
  // ignore: constant_identifier_names
  L2,
}
