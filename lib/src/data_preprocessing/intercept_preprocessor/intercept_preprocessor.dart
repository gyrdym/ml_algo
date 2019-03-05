import 'package:ml_linalg/matrix.dart';

abstract class InterceptPreprocessor {
  Matrix addIntercept(Matrix points);
}
