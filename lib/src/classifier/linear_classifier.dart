import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/matrix.dart';

/// An interface for all types of linear classifiers
abstract class LinearClassifier implements Classifier {
  /// A function that is used for converting learned coefficients into
  /// probabilities
  LinkFunction get linkFunction;

  /// A flag, that denotes, whether the intercept term considered during
  /// learning of the classifier or not
  bool get fitIntercept;

  /// A value, that defines a size of the intercept, if [fitIntercept] is
  /// `true`
  num get interceptScale;

  /// A matrix, where each column is a vector of coefficients, associated with
  /// the specific class
  Matrix get coefficientsByClasses;
}