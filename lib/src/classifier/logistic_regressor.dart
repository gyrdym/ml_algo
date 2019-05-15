import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/linear_classifier_mixin/linear_classifier_mixin.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/helpers/add_intercept.dart';
import 'package:ml_algo/src/helpers/get_batch_size.dart';
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
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressor with LinearClassifierMixin implements Classifier {
  LogisticRegressor(this.trainingFeatures, this.trainingOutcomes, {
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
    this.dtype = DefaultParameterValues.dtype,
    this.probabilityThreshold = 0.5,

    // private arguments
    ScoreToProbMapperFactory scoreToProbMapperFactory =
      const ScoreToProbMapperFactoryImpl(),

    OptimizerFactory optimizerFactory =
      const OptimizerFactoryImpl(),
  })  :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        scoreToProbMapper =
          scoreToProbMapperFactory.fromType(_scoreToProbMapperType, dtype),
        classLabels = trainingOutcomes.uniqueRows(),
        optimizer = optimizerFactory.fromType(
          optimizer,
          addInterceptIf(trainingFeatures, fitIntercept, interceptScale),
          trainingOutcomes,
          dtype: dtype,
          costFunctionType: CostFunctionType.logLikelihood,
          scoreToProbMapperType: _scoreToProbMapperType,
          learningRateType: learningRateType,
          initialWeightsType: initialWeightsType,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minWeightsUpdate,
          iterationLimit: iterationsLimit,
          lambda: lambda,
          batchSize: getBatchSize(gradientType, batchSize),
          randomSeed: randomSeed,
        ) {
    if (trainingOutcomes.columnsNum > 1) {
      throw Exception('Probably, you are trying to classify data with '
          'multiclass dependant variable, since the columns number of passed '
          'outcomes matrix is greater than 1 (${trainingOutcomes.columnsNum} '
          'given)');
    }
  }

  static const ScoreToProbMapperType _scoreToProbMapperType =
      ScoreToProbMapperType.logit;

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  @override
  final Matrix trainingFeatures;

  @override
  final Matrix trainingOutcomes;

  @override
  final Matrix classLabels;

  final DType dtype;

  final double probabilityThreshold;

  @override
  final Optimizer optimizer;

  @override
  final ScoreToProbMapper scoreToProbMapper;

  @override
  Matrix predictClasses(Matrix features) {
    final processedFeatures = addInterceptIf(trainingFeatures, fitIntercept,
        interceptScale);
    final classesSource = checkDataAndPredictProbabilities(processedFeatures)
        .getColumn(0)
    // TODO: use SIMD
        .map((value) => value >= probabilityThreshold ? 1.0 : 0.0)
        .toList(growable: false);
    return Matrix.fromColumns([Vector.fromList(classesSource)]);
  }
}
