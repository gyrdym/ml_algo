import 'dart:async';

import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class MLData<E> {
  Future<MLMatrix<E>> get features;
  Future<MLVector<E>> get labels;
}