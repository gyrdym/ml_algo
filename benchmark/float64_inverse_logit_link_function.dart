// 0.3 sec

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/link_function/logit/float64_inverse_logit_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const scoresCount = 100000;

class Float64InverseLogitLinkFunctionBenchmark extends BenchmarkBase {
  Float64InverseLogitLinkFunctionBenchmark() :
        super('Float64InverseLogitLinkFunctionBenchmark benchmark');

  final linkFunction = const Float64InverseLogitLinkFunction();

  Matrix samples;

  static void main() {
    Float64InverseLogitLinkFunctionBenchmark().report();
  }

  @override
  void run() {
    linkFunction.link(samples);
  }

  @override
  void setup() {
    final scores = Vector.randomFilled(scoresCount, min: -20, max: 20);

    samples = Matrix.fromColumns([
      scores,
    ]);
  }

  void tearDown() {}
}

Future main() async {
  Float64InverseLogitLinkFunctionBenchmark.main();
}
