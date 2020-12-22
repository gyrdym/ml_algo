import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_impl.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class SoftmaxRegressorFactoryImpl implements SoftmaxRegressorFactory {
  const SoftmaxRegressorFactoryImpl();

  @override
  SoftmaxRegressor create(
      LinearOptimizerType optimizerType,
      int iterationsLimit,
      double initialLearningRate,
      double minCoefficientsUpdate,
      double lambda,
      RegularizationType regularizationType,
      int randomSeed,
      int batchSize,
      bool isFittingDataNormalized,
      LearningRateType learningRateType,
      InitialCoefficientsType initialCoefficientsType,
      Matrix initialCoefficients,
      Matrix coefficientsByClasses,
      List<String> classNames,
      LinkFunction linkFunction,
      bool fitIntercept,
      num interceptScale,
      num positiveLabel,
      num negativeLabel,
      List<num> costPerIteration,
      DType dtype,
  ) => SoftmaxRegressorImpl(
    optimizerType,
    iterationsLimit,
    initialLearningRate,
    minCoefficientsUpdate,
    lambda,
    regularizationType,
    randomSeed,
    batchSize,
    isFittingDataNormalized,
    learningRateType,
    initialCoefficientsType,
    initialCoefficients,
    coefficientsByClasses,
    classNames,
    linkFunction,
    fitIntercept,
    interceptScale,
    positiveLabel,
    negativeLabel,
    costPerIteration,
    dtype,
  );
}
