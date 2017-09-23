part of 'package:dart_ml/src/interface.dart';

abstract class InitialWeightsGenerator {
  Float32x4Vector generate(int length);
}
