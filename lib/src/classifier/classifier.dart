import 'package:ml_algo/predictor.dart';
import 'package:ml_linalg/vector.dart';

abstract class Classifier implements Predictor {
  /// A map, where each key is a class label and each value, associated with the key, is a set of weights
  /// (coefficients), specific for the class
  Map<double, MLVector> get weightsByClasses;

  /// A collection of encoded class labels. Can be transformed back to original labels by a [MLData] instance, that was
  /// used previously to encode the labels
  Iterable<double> get classLabels;
}
