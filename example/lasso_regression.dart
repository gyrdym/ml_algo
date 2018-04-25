import 'dart:io';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

main() async {
  final csvCodec = new csv.CsvCodec();
  final input = new File('example/datasets/advertising.csv').openRead();
  final fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  final features = fields
      .map((List<num> item) => new Float32x4Vector.from(extractFeatures(item)))
      .toList(growable: false);

  final labels = fields.map((List<num> item) => item.last.toDouble()).toList();
  final lassoRegressionModel = new LassoRegressor(iterationLimit: 500, lambda: 74290.0);
  final validator = new CrossValidator<Float32x4Vector>.KFold();

  print('K-fold cross validation with MAPE metric:');
  print('Lasso regressor: ${validator.evaluate(lassoRegressionModel, features, labels, MetricType.MAPE)}');

  print('Feature weights (possibly, some weights are downgraded to zero, cause it is an objective of Lasso Regression):');
  print(lassoRegressionModel.weights);
}
