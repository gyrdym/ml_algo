import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_impl.dart';

class RandomizerFactoryImpl implements RandomizerFactory {
  const RandomizerFactoryImpl();

  @override
  Randomizer create([int seed]) => RandomizerImpl(seed: seed);
}
