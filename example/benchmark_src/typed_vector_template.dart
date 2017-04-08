part of dart_ml.benchmark.templates;

TypedVector typedVector1;
TypedVector typedVector2;

class TypedVectorInitBenchmark extends BenchmarkBase {
  const TypedVectorInitBenchmark() : super('Template');

  static void main() {
    new TypedVectorInitBenchmark().report();
  }

  void run() {
    typedVector1 = new TypedVector.fromList(new List<double>.filled(AMOUNT_OF_ELEMENTS, 1.0));
  }

  void tearDown() {
    typedVector1 = null;
  }
}

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

  void tearDown() {
    typedVector1 = null;
    typedVector2 = null;
  }
}