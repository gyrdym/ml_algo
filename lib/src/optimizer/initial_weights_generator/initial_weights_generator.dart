import 'package:linalg/vector.dart';

abstract class InitialWeightsGenerator<S extends List<E>, T extends List<double>, E> {
  SIMDVector<S, T, E> generate(int length);
}
