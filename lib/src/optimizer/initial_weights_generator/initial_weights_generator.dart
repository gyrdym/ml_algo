import 'package:linalg/vector.dart';

abstract class InitialWeightsGenerator<E> {
  Vector<E> generate(int length);
}
