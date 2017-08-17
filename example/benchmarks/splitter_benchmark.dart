import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:dart_ml/src/data_splitter/leave_p_out_impl.dart';

Iterable<Iterable<int>> result;

class LeavePOutSplitterBenchmark extends BenchmarkBase {
  const LeavePOutSplitterBenchmark() : super('Leave p out splitter test, p = 3');

  static void main() {
    new LeavePOutSplitterBenchmark().report();
    print('result is: $result');
  }

  void run() {
    LeavePOutSplitterImpl splitter = new LeavePOutSplitterImpl();
    splitter.configure(p: 3);
    result = splitter.split(100);
  }
}


main() {
  print('Measuring...');
  LeavePOutSplitterBenchmark.main();
}