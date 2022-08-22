import 'package:ml_linalg/vector.dart';

abstract class InitialCoefficientsGenerator {
  Vector generate(int length);
}
