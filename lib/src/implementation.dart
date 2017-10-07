import 'dart:math';
import 'dart:typed_data' show Float32List;

import 'package:di/di.dart';
import 'package:simd_vector/vector.dart';

import 'interface.dart';

part 'package:dart_ml/src/data_splitter/factory.dart';
part 'package:dart_ml/src/data_splitter/k_fold_impl.dart';
part 'package:dart_ml/src/data_splitter/leave_p_out_impl.dart';
part 'package:dart_ml/src/di/factory.dart';
part 'package:dart_ml/src/di/injector.dart';
part 'package:dart_ml/src/loss_function/cross_entropy.dart';
part 'package:dart_ml/src/loss_function/logistic_loss.dart';
part 'package:dart_ml/src/loss_function/loss_function_factory.dart';
part 'package:dart_ml/src/loss_function/squared_loss.dart';
part 'package:dart_ml/src/math/math.dart';
part 'package:dart_ml/src/math/math_analysis/gradient_calculator_impl.dart';
part 'package:dart_ml/src/math/randomizer/randomizer_impl.dart';
part 'package:dart_ml/src/metric/classification/accuracy.dart';
part 'package:dart_ml/src/metric/classification/metric_factory.dart';
part 'package:dart_ml/src/metric/regression/mape.dart';
part 'package:dart_ml/src/metric/regression/metric_factory.dart';
part 'package:dart_ml/src/metric/regression/rmse.dart';
part 'package:dart_ml/src/optimizer/gradient/base_impl.dart';
part 'package:dart_ml/src/optimizer/gradient/batch_impl.dart';
part 'package:dart_ml/src/optimizer/gradient/factory.dart';
part 'package:dart_ml/src/optimizer/gradient/initial_weights_generator/initial_weights_generator_factory.dart';
part 'package:dart_ml/src/optimizer/gradient/initial_weights_generator/zero_weights_generator.dart';
part 'package:dart_ml/src/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
part 'package:dart_ml/src/optimizer/gradient/learning_rate_generator/simple_learning_rate_generator.dart';
part 'package:dart_ml/src/optimizer/gradient/mini_batch_impl.dart';
part 'package:dart_ml/src/optimizer/gradient/stochastic_impl.dart';
part 'package:dart_ml/src/score_function/linear.dart';
part 'package:dart_ml/src/score_function/score_function_factory.dart';