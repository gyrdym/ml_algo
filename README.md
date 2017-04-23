# Machine learning with dart

Only linear regression is available now (with Batch, mini-Batch and Stochastic gradient descent optimizers)

## Usage

A simple usage example:

````dart  
import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

main() async {
  csv.CsvCodec csvCodec = new csv.CsvCodec();
  Stream<List<int>> input = new File('example/advertising.csv').openRead();
  List<List<num>> fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  List<TypedVector> allFeatures = fields
      .map((List<num> item) => new TypedVector.from(extractFeatures(item)))
      .toList(growable: false);

  TypedVector allLabels = new TypedVector.from(fields.map((List<num> item) => item.last.toDouble()).toList());

  Map<DataCategory, List<VectorInterface>> splittedFeatures = DataTrainTestSplitter.splitMatrix(allFeatures, .6);
  Map<DataCategory, VectorInterface> splitedLabels = DataTrainTestSplitter.splitVector(allLabels, .6);

  List<TypedVector> trainFeatures = splittedFeatures[DataCategory.TRAIN];
  TypedVector trainLabels = splitedLabels[DataCategory.TRAIN];

  List<TypedVector> testFeatures = splittedFeatures[DataCategory.TEST];
  TypedVector testLabels = splitedLabels[DataCategory.TEST];

  RMSEEstimator rmseEstimator = new RMSEEstimator();
  MAPEEstimator mapeEstimator = new MAPEEstimator();
  GradientLinearRegressor sgdRegressor = new GradientLinearRegressor<TypedVector, SGDOptimizer<TypedVector>>();
  
  sgdRegressor.train(trainFeatures, trainLabels);
  print("SGD regressor weights: ${sgdRegressor.weights}"););

  VectorInterface sgdPrediction = sgdRegressor.predict(testFeatures);
  print("SGD regressor, rmse (test) is: ${rmseEstimator.calculateError(sgdPrediction, testLabels)}");
  print("SGD regressor, mape (test) is: ${mapeEstimator.calculateError(sgdPrediction, testLabels)}\n");
}

````