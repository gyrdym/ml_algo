import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'package:web/web.dart' as web;

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

class FooApi {
  @JSExport('decisionTreeFull')
  String decisionTreeFull(int n, int n2, int n3, int n4) =>
      _decisionTreeFull(n, n2, n3, n4);

  @JSExport('decisionTreeLoaded')
  String decisionTreeLoaded(int n, int n2, int n3, int n4) =>
      _decisionTreeLoaded(n, n2, n3, n4);

  String _decisionTreeLoaded(int n, int n2, int n3, int n4) {
    if (n == 0 && n2 == 0 && n3 == 0 && n4 == 0) {
      return 'unknown';
    }
    final model = DecisionTreeClassifier.fromJson(modelJson);
    final d = DataFrame([
      [n, n2, n3, n4]
    ], headerExists: false);
    final dfResult = model.predict(d);
    final result = dfResult.rows.first.first;
    if (result == 0.0) {
      return 'Iris-setosa';
    } else if (result == 1.0) {
      return 'Iris-versicolor';
    } else if (result == 2.0) {
      return 'Iris-virginica';
    } else {
      return 'unknown';
    }
  }

  String _decisionTreeFull(int n, int n2, int n3, int n4) {
    if (n == 0 && n2 == 0 && n3 == 0 && n4 == 0) {
      return 'unknown';
    }
    final samples = getIrisDataFrame().shuffle().dropSeries(names: ['Id']);
    final pipeline = Pipeline(samples, [
      // Here we convert strings from 'Species' column into numbers
      toIntegerLabels(columnNames: ['Species']),
    ]);
    final processed = pipeline.process(samples);
    final model = DecisionTreeClassifier(
      processed,
      'Species',
      minError: 0.3,
      minSamplesCount: 5,
      maxDepth: 4,
    );

//  getIrisDataFrame().rows;
    final d = DataFrame([
      [n, n2, n3, n4],
    ], headerExists: false);
    final dfResult = model.predict(d);
    final result = dfResult.rows.first.first;
    if (result == 0.0) {
      return 'Iris-setosa';
    } else if (result == 1.0) {
      return 'Iris-versicolor';
    } else if (result == 2.0) {
      return 'Iris-virginica';
    } else {
      return 'unknown';
    }
  }
}

void main() {
  // One line to export the `FooApi`.
  web.window.setProperty('foo'.toJS, createJSInteropWrapper(FooApi()));
}

const modelJson =
    '''{"E":0.3,"S":5,"D":4,"DT":"F32","T":"Species","R":{"CN":[{"LB":{"V":0.0,"P":1.0000000000000004},"PT":"LT","SV":2.649999976158142,"SI":2},{"CN":[{"LB":{"V":2.0,"P":0.9074074074074064},"PT":"LT","SV":1.7499999403953552,"SI":3},{"LB":{"V":1.0,"P":0.9782608695652169},"PT":"GET","SV":1.7499999403953552,"SI":3}],"PT":"GET","SV":2.649999976158142,"SI":2}]},"\$V":1,"A":"G"}''';
