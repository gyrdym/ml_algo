import 'benchmark_src/benchmarks.dart';

main() {
  print('Measuring...');
  TypedVectorInitBenchmark.main();
  TypedVectorAdditionBenchmark.main();
  RegularVectorInitBenchmark.main();
  RegularListVectorAdditionBenchmark.main();
}
