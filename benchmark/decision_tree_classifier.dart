// MacBook Air 13.3 mid 2017: ~ 6.5 sec

import 'dart:math';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';

const observationsNum = 300;
const columnsNum = 11;

class DecisionTreeClassifierBenchmark extends BenchmarkBase {
  DecisionTreeClassifierBenchmark()
      : super('Decision tree classifier benchmark');

  late DataFrame _data;

  static void main() {
    DecisionTreeClassifierBenchmark().report();
  }

  @override
  void run() {
    DecisionTreeClassifier(
      _data,
      'col_10',
      maxDepth: 4,
      minError: 0.4,
      minSamplesCount: 10,
    );
  }

  @override
  void setup() {
    final random = Random(1);
    final observations =
        Matrix.random(observationsNum, columnsNum - 1, seed: 1);
    final outcomes = Vector.fromList([
      ...List.filled(observationsNum ~/ 2, 1),
      ...List.filled(observationsNum ~/ 2, 0)
    ]..shuffle(random));

    _data = DataFrame(observations.insertColumns(columnsNum - 1, [outcomes]),
        headerExists: false);
  }

  void tearDown() {}
}

Future main() async {
  DecisionTreeClassifierBenchmark.main();
}
