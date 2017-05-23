import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'package:csv/csv.dart' as csv;
import 'package:dart_ml/dart_ml.dart';
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:dart_ml/src/validators/kfold_cross_validator.dart';

SGDLinearRegressor predictor;
List<TypedVector> features;
TypedVector labels;

class KFoldValidatorBenchmark extends BenchmarkBase {
  const KFoldValidatorBenchmark() : super('k-fold validator test');

  static void main() {
    new KFoldValidatorBenchmark().report();
  }

  void run() {
    KFoldCrossValidator validator = new KFoldCrossValidator();
    validator.validate(predictor, features, labels);
  }
}

Future main() async {
  csv.CsvCodec csvCodec = new csv.CsvCodec();
  Stream<List<int>> input = new File('example/advertising.csv').openRead();
  List<List<num>> fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  features = fields
      .map((List<num> item) => new TypedVector.from(extractFeatures(item)))
      .toList(growable: false);

  labels = new TypedVector.from(fields.map((List<num> item) => item.last.toDouble()).toList());

  predictor = new SGDLinearRegressor();

  print('measuring...');
  KFoldValidatorBenchmark.main();
}
