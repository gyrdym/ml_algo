import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:dart_ml/src/data_splitters/leave_p_out_splitter.dart';

Iterable<Iterable<int>> result;

class LeavePOutSplitterBenchmark extends BenchmarkBase {
  const LeavePOutSplitterBenchmark() : super('Leave p out splitter test, p = 3');

  static void main() {
    new LeavePOutSplitterBenchmark().report();
    print('result is: $result');
  }

  void run() {
    LeavePOutSplitter splitter = new LeavePOutSplitter(p: 3);
    result = splitter.split(100);
  }
}
