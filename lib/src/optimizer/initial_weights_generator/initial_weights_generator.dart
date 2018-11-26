import 'package:linalg/linalg.dart';

abstract class InitialWeightsGenerator<E> {
  Vector<E> generate(int length);
}
