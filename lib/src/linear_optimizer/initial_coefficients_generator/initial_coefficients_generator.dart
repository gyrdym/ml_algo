import 'package:ml_linalg/linalg.dart';

abstract class InitialCoefficientsGenerator {
  Vector generate(int length);
}
