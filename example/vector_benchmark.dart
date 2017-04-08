import 'benchmark_src/benchmark_templates.dart';

main() {
  print('Typed vector initialization ($AMOUNT_OF_ELEMENTS elements)...');
  TypedVectorInitBenchmark.main();

  print('\n');

  print('Typed vectors addition ($AMOUNT_OF_ELEMENTS elements)...');
  TypedVectorAdditionBenchmark.main();

  print('\n');

  print('Regular list-based vector initialization ($AMOUNT_OF_ELEMENTS elements)...');
  RegularListVectorInitBenchmark.main();

  print('\n');

  print('Regular list-based vectors addition ($AMOUNT_OF_ELEMENTS elements)...');
  RegularListVectorAdditionBenchmark.main();
}
