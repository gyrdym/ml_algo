import 'package:ml_linalg/matrix.dart';

abstract class InterceptPreprocessor {
  MLMatrix addIntercept(MLMatrix points);
}
