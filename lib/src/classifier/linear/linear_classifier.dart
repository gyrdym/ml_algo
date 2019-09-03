import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_linalg/matrix.dart';

abstract class LinearClassifier implements Classifier {
  /// A matrix, where each column is a vector of coefficients, associated with
  /// the specific class
  Matrix get coefficientsByClasses;
}