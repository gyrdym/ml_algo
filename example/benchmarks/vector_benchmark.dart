import 'vector/benchmarks.dart';

main() {
  print('Measuring...');
  TypedVectorInitBenchmark.main();
  TypedVectorAdditionBenchmark.main();
  RegularVectorInitBenchmark.main();
  RegularListVectorAdditionBenchmark.main();
}
