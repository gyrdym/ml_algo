import 'package:ml_linalg/linalg.dart';

abstract class InitialWeightsGenerator {
  MLVector generate(int length);
}
