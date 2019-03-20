import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';

class OneHotEncoderBenchmark extends BenchmarkBase {
  OneHotEncoderBenchmark(this._encoder, this._data)
      : super('One Hot Encoder benchmark');

  final OneHotEncoder _encoder;
  final Iterable<String> _data;

  @override
  void run() {
    _encoder.encode(_data);
  }
}

void oneHotEncoderBenchmark() {
  final numOfLabels = 10000;
  final encoder = OneHotEncoder();
  OneHotEncoderBenchmark(encoder,
      List<String>.generate(numOfLabels, (int idx) => 'label_$idx'))
    ..report();
}
