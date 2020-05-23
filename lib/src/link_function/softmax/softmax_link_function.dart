import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/matrix.dart';

abstract class SoftmaxLinkFunction implements LinkFunction {
  Matrix getNumerator(Matrix scores);
}
