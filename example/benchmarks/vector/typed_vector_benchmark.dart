part of dart_ml.benchmark;

Vector typedVector1;
Vector typedVector2;

class TypedVectorInitBenchmark extends BenchmarkBase {
  const TypedVectorInitBenchmark() : super('Typed vector initialization, $AMOUNT_OF_ELEMENTS elements');

  static void main() {
    new TypedVectorInitBenchmark().report();
  }

  void run() {
    typedVector1 = new Vector.from(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
  }

  void tearDown() {
    typedVector1 = null;
  }
}

class TypedVectorAdditionBenchmark extends BenchmarkBase {
  const TypedVectorAdditionBenchmark() : super('Typed vectors addition, $AMOUNT_OF_ELEMENTS elements');

  static void main() {
    new TypedVectorAdditionBenchmark().report();
  }

  void run() {
    typedVector1 + typedVector2;
  }

  void setup() {
    typedVector1 = new Vector.from(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
    typedVector2 = new Vector.from(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
  }

  void tearDown() {
    typedVector1 = null;
    typedVector2 = null;
  }
}