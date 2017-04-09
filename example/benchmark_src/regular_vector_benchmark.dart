part of dart_ml.benchmark;

RegularVector regularVector1;
RegularVector regularVector2;

class RegularVectorInitBenchmark extends BenchmarkBase {
  const RegularVectorInitBenchmark() : super('Regular list-based vector initialization, $AMOUNT_OF_ELEMENTS elements');

  static void main() {
    new RegularVectorInitBenchmark().report();
  }

  void run() {
    regularVector1 = new RegularVector.fromList(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
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
    regularVector1 + regularVector2;
  }

  void setup() {
    regularVector1 = new RegularVector.fromList(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
    regularVector2 = new RegularVector.fromList(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
  }

  void tearDown() {
    regularVector1 = null;
    regularVector2 = null;
  }
}