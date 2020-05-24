// 0.24 sec

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const scoresCount = 100000;

class Float32InverseLogitLinkFunctionBenchmark extends BenchmarkBase {
  Float32InverseLogitLinkFunctionBenchmark() :
        super('Float32InverseLogitLinkFunction benchmark');

  final linkFunction = const Float32InverseLogitLinkFunction();

  Matrix samples;

  static void main() {
    Float32InverseLogitLinkFunctionBenchmark().report();
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
  Float32InverseLogitLinkFunctionBenchmark.main();
}
