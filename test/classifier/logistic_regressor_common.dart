import 'dart:typed_data';

import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';

import '../test_utils/mocks.dart';

LabelsProcessor labelsProcessorMock;
LabelsProcessorFactory labelsProcessorFactoryMock;
InterceptPreprocessor interceptPreprocessorMock;
InterceptPreprocessorFactory interceptPreprocessorFactoryMock;
Optimizer optimizerMock;
OptimizerFactory optimizerFactoryMock;
LinkFunctionFactory linkFunctionFactoryMock;
LinkFunction linkFunctionMock;

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

void setUpLinkFunctionFactory() {
  linkFunctionMock = LinkFunctionMock();
  linkFunctionFactoryMock =
      createLinkFunctionFactoryMock(Float32x4, linkFunctions: {
    LinkFunctionType.logit: linkFunctionMock,
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
      linkFunctionType: LinkFunctionType.logit,
      linkFunctionFactory: linkFunctionFactoryMock,
      optimizer: OptimizerType.gradientDescent,
      optimizerFactory: optimizerFactoryMock,
      gradientType: GradientType.stochastic,
      randomSeed: randomSeed,
    );
