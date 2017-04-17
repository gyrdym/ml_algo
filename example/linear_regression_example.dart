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

  SGDLinearRegressor predictor = new SGDLinearRegressor<TypedVector>();

  List<TypedVector> allFeatures = fields
      .map((List<num> item) => new TypedVector.from(extractFeatures(item)))
      .toList(growable: false);

  TypedVector allLabels = new TypedVector.from(fields.map((List<num> item) => item.last.toDouble()).toList());

  Map<String, List<VectorInterface>> splittedFeatures = splitMatrix(allFeatures, .6);
  Map<String, VectorInterface> splitedLabels = splitVector(allLabels, .6);

  List<TypedVector> trainFeatures = splittedFeatures['train'];
  TypedVector trainLabels = splitedLabels['train'];

  List<TypedVector> testFeatures = splittedFeatures['test'];
  TypedVector testLabels = splitedLabels['test'];

  predictor.train(trainFeatures, trainLabels);
  print("weights: ${predictor.weights}");

  VectorInterface prediction = predictor.predict(testFeatures);
  print("rmse (test) is: ${predictor.estimator.calculateError(prediction, testLabels)}");
}

Map<String, List<VectorInterface>> splitMatrix(List<VectorInterface> sample, double trainRatio) {
  int ratioLength = (sample.length * trainRatio).floor();

  return <String, List<VectorInterface>>{
    'train': sample.sublist(0, ratioLength),
    'test': sample.sublist(ratioLength)
  };
}

Map<String, VectorInterface> splitVector(VectorInterface sample, double trainRatio) {
  int ratioLength = (sample.length * trainRatio).floor();

  return <String, VectorInterface>{
    'train': sample.fromRange(0, ratioLength),
    'test': sample.fromRange(ratioLength)
  };
}
