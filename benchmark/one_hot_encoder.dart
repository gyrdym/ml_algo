import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';

class OneHotEncoderBenchmark extends BenchmarkBase {
  final OneHotEncoder _encoder;
  final Iterable<Object> _data;

  OneHotEncoderBenchmark(this._encoder, this._data)
      : super('One Hot Encoder benchmark');

  @override
  void run() {
    _data.forEach(_encoder.encodeSingle);
  }
}

void oneHotEncoderBenchmark() {
  final numOfLabels = 10000;
  final encoder = OneHotEncoder()
    ..setCategoryValues(
        List<String>.generate(numOfLabels, (int idx) => 'label_$idx'));
  OneHotEncoderBenchmark(encoder, ['label_100', 'label_200', 'label_300'])
    ..report();
}
