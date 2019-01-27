import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_linalg/linalg.dart';

class ZeroWeightsGenerator<T> implements InitialWeightsGenerator<T> {
  @override
  MLVector<T> generate(int length) => MLVector<T>.from(List.filled(length, 0.0));
}
