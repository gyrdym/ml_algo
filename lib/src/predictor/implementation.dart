import 'dart:typed_data' show Float32List;

import 'package:dart_ml/src/core/implementation.dart';
import 'package:dart_ml/src/core/interface.dart';
import 'package:di/di.dart';
import 'package:simd_vector/vector.dart' show Float32x4Vector;

import 'interface.dart';

part 'package:dart_ml/src/predictor/base/classifier_base.dart';
part 'package:dart_ml/src/predictor/base/predictor_base.dart';
part 'package:dart_ml/src/predictor/linear/classifier/gradient/logistic_regression.dart';
part 'package:dart_ml/src/predictor/linear/regressor/gradient/batch.dart';
part 'package:dart_ml/src/predictor/linear/regressor/gradient/mini_batch.dart';
part 'package:dart_ml/src/predictor/linear/regressor/gradient/stochastic.dart';