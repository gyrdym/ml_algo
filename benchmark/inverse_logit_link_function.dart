//  Approx. 2.5 sec (Macbook Pro 2019, Dart 2.16.0)

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const scoresCount = 2000000;

class InverseLogitLinkFunctionBenchmark extends BenchmarkBase {
  InverseLogitLinkFunctionBenchmark()
      : super('InverseLogitLinkFunctionBenchmark benchmark');

  final linkFunction = const InverseLogitLinkFunction();

  late Matrix samples;

  static void main() {
    InverseLogitLinkFunctionBenchmark().report();
  }

  @override
  void run() {
    linkFunction.link(samples);
  }

  @override
  void setup() {
    final scores = Vector.randomFilled(scoresCount, min: -20, max: 20, seed: 12);

    samples = Matrix.fromColumns([
      scores,
    ]);
  }

  void tearDown() {}
}

Future<void> main() async {
  InverseLogitLinkFunctionBenchmark.main();
}
