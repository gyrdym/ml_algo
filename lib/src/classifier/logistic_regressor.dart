import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/linear_classifier_mixin/linear_classifier_mixin.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/optimizer/gradient/batch_size_calculator/batch_size_calculator.dart';
import 'package:ml_algo/src/optimizer/gradient/batch_size_calculator/batch_size_calculator_impl.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory_impl.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_algo/src/regressor/gradient_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory_impl.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressor with LinearClassifierMixin implements Classifier {
  LogisticRegressor({
    // public arguments
    int iterationsLimit = DefaultParameterValues.iterationsLimit,
    double initialLearningRate = DefaultParameterValues.initialLearningRate,
    double minWeightsUpdate = DefaultParameterValues.minCoefficientsUpdate,
    double lambda,
    int randomSeed,
    int batchSize = 1,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    OptimizerType optimizer = OptimizerType.gradientDescent,
    GradientType gradientType = GradientType.stochastic,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    ScoreToProbMapperType scoreToProbMapperType = ScoreToProbMapperType.logit,
    this.dtype = DefaultParameterValues.dtype,

    // private arguments
    InterceptPreprocessorFactory interceptPreprocessorFactory =
      const InterceptPreprocessorFactoryImpl(),

    ScoreToProbMapperFactory scoreToProbMapperFactory =
      const ScoreToProbMapperFactoryImpl(),

    OptimizerFactory optimizerFactory =
      const OptimizerFactoryImpl(),

    BatchSizeCalculator batchSizeCalculator =
      const BatchSizeCalculatorImpl(),
  })  : interceptPreprocessor = interceptPreprocessorFactory.create(dtype,
            scale: fitIntercept ? interceptScale : 0.0),
        scoreToProbMapper =
            scoreToProbMapperFactory.fromType(scoreToProbMapperType, dtype),
        optimizer = optimizerFactory.fromType(
          optimizer,
          dtype: dtype,
          costFunctionType: CostFunctionType.logLikelihood,
          scoreToProbMapperType: scoreToProbMapperType,
          learningRateType: learningRateType,
          initialWeightsType: initialWeightsType,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minWeightsUpdate,
          iterationLimit: iterationsLimit,
          lambda: lambda,
          batchSize: gradientType != null
              ? batchSizeCalculator.calculate(gradientType, batchSize)
              : null,
          randomSeed: randomSeed,
        );

  @override
  final Type dtype;

  @override
  final Optimizer optimizer;

  @override
  final InterceptPreprocessor interceptPreprocessor;

  @override
  final ScoreToProbMapper scoreToProbMapper;

  /// [labels] - either matrix-column of 0 and 1 or one-vs-all encoded
  /// multiclass label list (m*n matrix, where m - rows number, n - number of
  /// classes)
  @override
  Matrix learnWeights(Matrix features, Matrix labels,
      Matrix initialWeights, bool arePointsNormalized) =>
    labels.mapColumns((column) => _fitBinaryClassifier(features, column,
        initialWeights, arePointsNormalized));

  Vector _fitBinaryClassifier(Matrix features, Vector labels,
      Matrix initialWeights, bool arePointsNormalized) =>
      optimizer
        .findExtrema(features, Matrix.columns([labels]),
            initialWeights: initialWeights,
            arePointsNormalized: arePointsNormalized,
            isMinimizingObjective: false)
        .getColumn(0);
}
