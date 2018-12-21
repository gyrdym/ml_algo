import 'dart:typed_data';

import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/cross_validator/float32x4_cross_validator.dart';

abstract class Float32x4CrossValidator {
  static CrossValidator<Float32x4> kFold({int numberOfFolds = 5}) =>
      Float32x4CrossValidatorInternal.kFold(numberOfFolds: numberOfFolds);

  static CrossValidator<Float32x4> lpo({int p = 5}) =>
      Float32x4CrossValidatorInternal.lpo(p: p);
}
