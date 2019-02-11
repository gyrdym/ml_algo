import 'dart:typed_data';

import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';

import '../test_utils/mocks.dart';

LabelsProcessor labelsProcessorMock;
LabelsProcessorFactory labelsProcessorFactoryMock;
InterceptPreprocessor interceptPreprocessorMock;
InterceptPreprocessorFactory interceptPreprocessorFactoryMock;
Optimizer optimizerMock;
OptimizerFactory optimizerFactoryMock;
ScoreToProbMapperFactory scoreToProbFactoryMock;
ScoreToProbMapper scoreToProbMapperMock;

void setUpLabelsProcessorFactory() {
  labelsProcessorMock = LabelsProcessorMock();
  labelsProcessorFactoryMock = createLabelsProcessorFactoryMock(
      processors: {Float32x4: labelsProcessorMock});
}

void setUpInterceptPreprocessorFactory() {
  interceptPreprocessorMock = InterceptPreprocessorMock();
  interceptPreprocessorFactoryMock = createInterceptPreprocessorFactoryMock(
      preprocessor: interceptPreprocessorMock);
}

void setUpOptimizerFactory() {
  optimizerMock = OptimizerMock();
  optimizerFactoryMock = createOptimizerFactoryMock(
      optimizers: {OptimizerType.gradientDescent: optimizerMock});
}

void setUpScoreToProbMapperFactory() {
  scoreToProbMapperMock = ScoreToProbMapperMock();
  scoreToProbFactoryMock =
      createScoreToProbMapperFactoryMock(Float32x4, mappers: {
    ScoreToProbMapperType.logit: scoreToProbMapperMock,
  });
}

LogisticRegressor createRegressor({
  int iterationLimit = 100,
  double learningRate = 0.01,
  double minWeightsUpdate = 0.001,
  double lambda = 0.1,
  int randomSeed = 123,
}) =>
    LogisticRegressor(
      dtype: Float32x4,
      learningRateType: LearningRateType.constant,
      initialWeightsType: InitialWeightsType.zeroes,
      iterationsLimit: iterationLimit,
      initialLearningRate: learningRate,
      minWeightsUpdate: minWeightsUpdate,
      lambda: lambda,
      labelsProcessorFactory: labelsProcessorFactoryMock,
      interceptPreprocessorFactory: interceptPreprocessorFactoryMock,
      scoreToProbMapperType: ScoreToProbMapperType.logit,
      scoreToProbMapperFactory: scoreToProbFactoryMock,
      optimizer: OptimizerType.gradientDescent,
      optimizerFactory: optimizerFactoryMock,
      gradientType: GradientType.stochastic,
      randomSeed: randomSeed,
    );
