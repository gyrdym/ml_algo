import 'package:ml_linalg/matrix.dart';

abstract class InterceptPreprocessor<T> {
  MLMatrix<T> addIntercept(MLMatrix<T> points);
}
