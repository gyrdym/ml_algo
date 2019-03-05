import 'package:ml_linalg/linalg.dart';

abstract class InitialWeightsGenerator {
  Vector generate(int length);
}
