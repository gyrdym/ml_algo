import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/matrix.dart';

abstract class LinearClassifier implements Classifier {
  LinkFunction get linkFunction;

  bool get fitIntercept;

  double get interceptScale;

  /// A matrix, where each column is a vector of coefficients, associated with
  /// the specific class
  Matrix get coefficientsByClasses;
}