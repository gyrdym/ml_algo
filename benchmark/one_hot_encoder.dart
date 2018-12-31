import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';

class OneHotEncoderBenchmark extends BenchmarkBase {
  final OneHotEncoder _encoder;
  final Map<String, Iterable<Object>> _data;

  OneHotEncoderBenchmark(this._encoder, this._data) : super('One Hot Encoder benchmark');

  @override
  void run() {
    _data.forEach((String label, Iterable<Object> data) {
      data.forEach((value) {
        _encoder.encode(label, value);
      });
    });
  }
}

void oneHotEncoderBenchmark() {
  final numOfLabels = 10000;
  final encoder = OneHotEncoder({
    'category': List<String>.generate(numOfLabels, (int idx) => 'label_$idx'),
  });
  OneHotEncoderBenchmark(encoder, {'category': ['label_100']})
    ..report();
}