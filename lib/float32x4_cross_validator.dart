import 'dart:typed_data';

import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/cross_validator/float32x4_cross_validator.dart';

abstract class Float32x4CrossValidator implements CrossValidator<Float32x4> {
  /// Creates k-fold validator to evaluate quality of a predictor. It splits a dataset into [numberOfFolds] test sets
  /// and subsequently evaluates the predictor on each produced test set
  factory Float32x4CrossValidator.kFold({int numberOfFolds}) = Float32x4CrossValidatorInternal.kFold;

  /// Creates LPO validator to evaluate quality of a predictor. It splits a dataset into all possible test sets of
  /// size [p] and subsequently evaluates quality of the predictor on each produced test set
  factory Float32x4CrossValidator.lpo({int p}) = Float32x4CrossValidatorInternal.lpo;
}
