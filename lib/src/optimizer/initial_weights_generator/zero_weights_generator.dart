import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_linalg/linalg.dart';

class ZeroWeightsGenerator implements InitialWeightsGenerator {
  @override
  MLVector generate(int length) => MLVector.from(List.filled(length, 0.0));
}
