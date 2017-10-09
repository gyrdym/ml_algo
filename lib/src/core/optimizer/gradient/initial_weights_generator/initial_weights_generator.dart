part of 'package:dart_ml/src/core/interface.dart';

abstract class InitialWeightsGenerator {
  Float32x4Vector generate(int length);
}
