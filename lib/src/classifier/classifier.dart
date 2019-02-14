import 'package:ml_algo/src/predictor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// An interface for any classifier (linear, non-linear, parametric, non-parametric, etc.)
abstract class Classifier implements Predictor {
  /// A map, where each key is a class label and each value, associated with the key, is a set of weights
  /// (coefficients), specific for the class
  Map<double, MLVector> get weightsByClasses;

  /// A collection of encoded class labels. Can be transformed back to original labels by a [MLData] instance, that was
  /// used previously to encode the labels
  Iterable<double> get classLabels;

  /// Returns predicted distribution of probabilities for each observation in the passed [features]
  MLMatrix predictProbabilities(MLMatrix features);

  /// Return a collection of predicted class labels for each observation in the passed [features]
  MLVector predictClasses(MLMatrix features);
}
