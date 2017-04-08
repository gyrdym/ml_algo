part of dart_ml.benchmark;

List<double> regularVector1;
List<double> regularVector2;

class RegularListVectorInitBenchmark extends BenchmarkBase {
  const RegularListVectorInitBenchmark() : super('Regular list-based vector initialization, $AMOUNT_OF_ELEMENTS elements');

  static void main() {
    new RegularListVectorInitBenchmark().report();
  }

  void run() {
    regularVector1 = new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0);
  }

  void tearDown() {
    regularVector1 = null;
  }
}

class RegularListVectorAdditionBenchmark extends BenchmarkBase {
  const RegularListVectorAdditionBenchmark() : super('Regular list-based vectors addition, $AMOUNT_OF_ELEMENTS elements');

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

  void tearDown() {
    regularVector1 = null;
    regularVector2 = null;
  }
}