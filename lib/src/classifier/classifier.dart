import 'package:ml_linalg/matrix.dart';

/// An interface for any classifier (linear, non-linear, parametric,
/// non-parametric, etc.)
abstract class Classifier {
  /// A collection of encoded class labels
  Matrix get classLabels;

  /// Returns predicted distribution of probabilities for each observation in
  /// the passed [features]
  Matrix predictProbabilities(Matrix features);

  /// Return a collection of predicted class labels for each observation in the
  /// passed [features]
  Matrix predictClasses(Matrix features);
}
