import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory_impl.dart';
import 'package:ml_algo/src/classifier/linear_classifier.dart';
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

class LogisticRegressor with LinearClassifierMixin implements LinearClassifier {
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
    LabelsProcessorFactory labelsProcessorFactory =
        const LabelsProcessorFactoryImpl(),
    InterceptPreprocessorFactory interceptPreprocessorFactory =
        const InterceptPreprocessorFactoryImpl(),
    ScoreToProbMapperFactory scoreToProbMapperFactory =
        const ScoreToProbMapperFactoryImpl(),
    OptimizerFactory optimizerFactory = const OptimizerFactoryImpl(),
    BatchSizeCalculator batchSizeCalculator = const BatchSizeCalculatorImpl(),
  })  : labelsProcessor = labelsProcessorFactory.create(dtype),
        interceptPreprocessor = interceptPreprocessorFactory.create(dtype,
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
  final LabelsProcessor labelsProcessor;

  @override
  final ScoreToProbMapper scoreToProbMapper;

  @override
  MLMatrix learnWeights(MLMatrix features, MLVector labels,
      MLMatrix initialWeights, bool arePointsNormalized) {
    final _weightsSource = List<MLVector>.generate(classLabels.length,
            (int i) => _fitBinaryClassifier(features, labels,
            classLabels[i], initialWeights, arePointsNormalized));
    return MLMatrix.columns(_weightsSource);
  }

  MLVector _fitBinaryClassifier(MLMatrix features, MLVector labels,
      double targetLabel, MLMatrix initialWeights, bool arePointsNormalized) {
    final binaryLabels = labelsProcessor.makeLabelsOneVsAll(labels,
        targetLabel);
    return optimizer
        .findExtrema(features, MLMatrix.columns([binaryLabels]),
            initialWeights: initialWeights?.transpose(),
            arePointsNormalized: arePointsNormalized,
            isMinimizingObjective: false)
        .getColumn(0);
  }
}
