import 'dart:typed_data';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
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

InterceptPreprocessor interceptPreprocessorMock;
InterceptPreprocessorFactory interceptPreprocessorFactoryMock;
Optimizer optimizerMock;
OptimizerFactory optimizerFactoryMock;
ScoreToProbMapperFactory scoreToProbFactoryMock;
ScoreToProbMapper scoreToProbMapperMock;
CategoricalDataEncoderFactory categoricalDataEncoderFactoryMock;
CategoricalDataEncoder categoricalDataEncoderMock;

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

void setUpCategoricalDataEncoderFactory() {
  categoricalDataEncoderMock = CategoricalDataEncoderMock();
  categoricalDataEncoderFactoryMock = createCategoricalDataEncoderFactoryMock(
      oneHotEncoderMock: categoricalDataEncoderMock);
}

LogisticRegressor createLogisticRegressor({
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
      interceptPreprocessorFactory: interceptPreprocessorFactoryMock,
      scoreToProbMapperType: ScoreToProbMapperType.logit,
      scoreToProbMapperFactory: scoreToProbFactoryMock,
      optimizer: OptimizerType.gradientDescent,
      optimizerFactory: optimizerFactoryMock,
      gradientType: GradientType.stochastic,
      randomSeed: randomSeed,
    );

SoftmaxRegressor createSoftmaxRegressor({
  int iterationLimit = 100,
  double learningRate = 0.01,
  double minWeightsUpdate = 0.001,
  double lambda = 0.1,
  int randomSeed = 123,
  Type dtype = Float32x4,
}) =>
    SoftmaxRegressor(
      dtype: dtype,
      learningRateType: LearningRateType.constant,
      initialWeightsType: InitialWeightsType.zeroes,
      iterationsLimit: iterationLimit,
      initialLearningRate: learningRate,
      minWeightsUpdate: minWeightsUpdate,
      lambda: lambda,
      interceptPreprocessorFactory: interceptPreprocessorFactoryMock,
      scoreToProbMapperType: ScoreToProbMapperType.logit,
      scoreToProbMapperFactory: scoreToProbFactoryMock,
      categoricalDataEncoderFactory: categoricalDataEncoderFactoryMock,
      optimizer: OptimizerType.gradientDescent,
      optimizerFactory: optimizerFactoryMock,
      gradientType: GradientType.stochastic,
      randomSeed: randomSeed,
    );
