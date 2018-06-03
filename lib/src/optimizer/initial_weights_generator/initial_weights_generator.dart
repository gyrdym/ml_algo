import 'package:linalg/vector.dart';

abstract class InitialWeightsGenerator {
  Vector generate(int length);
}
