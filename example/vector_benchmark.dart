import 'benchmark_src/benchmarks.dart';

main() {
  print('Measuring...');
  TypedVectorInitBenchmark.main();
  TypedVectorAdditionBenchmark.main();
  RegularListVectorInitBenchmark.main();
  RegularListVectorAdditionBenchmark.main();
}
