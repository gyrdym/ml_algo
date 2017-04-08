part of dart_ml.benchmark.templates;

List<double> regularVector1;
List<double> regularVector2;

class RegularListVectorInitBenchmark extends BenchmarkBase {
  const RegularListVectorInitBenchmark() : super('Template');

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

  void tearDown() {
    regularVector1 = null;
    regularVector2 = null;
  }
}