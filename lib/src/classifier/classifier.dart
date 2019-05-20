import 'package:ml_linalg/matrix.dart';

/// An interface for any classifier (linear, non-linear, parametric,
/// non-parametric, etc.)
abstract class Classifier {
  /// A matrix, where each column is a vector of weights, associated with
  /// the specific class
  Matrix get weightsByClasses;

  /// A collection of class labels. Can be transformed back to original
  /// labels by a [MLData] instance, that was used previously to encode the
  /// labels
  Matrix get classLabels;

  /// Returns predicted distribution of probabilities for each observation in
  /// the passed [features]
  Matrix predictProbabilities(Matrix features);
}
