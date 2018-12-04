import 'package:ml_linalg/linalg.dart';

abstract class InitialWeightsGenerator<E> {
  MLVector<E> generate(int length);
}
