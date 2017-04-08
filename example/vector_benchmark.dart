import 'package:dart_ml/dart_ml.dart';
import 'package:benchmark_harness/benchmark_harness.dart';

const int AMOUNT_OF_ELEMENTS = 10000000;

TypedVector typedVector1;
TypedVector typedVector2;

List<double> regularVector1;
List<double> regularVector2;

class TypedVectorAdditionBenchmark extends BenchmarkBase {
  const TypedVectorAdditionBenchmark() : super('Template');

  static void main() {
    new TypedVectorAdditionBenchmark().report();
  }

  void run() {
    typedVector1 + typedVector2;
  }

  void setup() {
    typedVector1 = new TypedVector.fromList(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
    typedVector2 = new TypedVector.fromList(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
  }

  void teardown() {}
}


class RegularListVectorAdditionBenchmark extends BenchmarkBase {
  const RegularListVectorAdditionBenchmark() : super('Template');

  static void main() {
    new RegularListVectorAdditionBenchmark().report();
  }

  void run() {
    vectorAddition(regularVector1, regularVector2);
  }

  void setup() {
    regularVector1 = new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0);
    regularVector2 = new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0);
  }

  void teardown() {}
}

main() {
  print('Typed vectors addition ($AMOUNT_OF_ELEMENTS elements)...');
  TypedVectorAdditionBenchmark.main();

  print('\n');

  print('Regular list-based vectors addition ($AMOUNT_OF_ELEMENTS elements)...');
  RegularListVectorAdditionBenchmark.main();
}
