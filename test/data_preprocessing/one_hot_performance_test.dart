import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:test/test.dart';

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

void main() {
  final numOfLabels = 10000;

  group('OneHotEncoder (performance)', () {
    test('should encode data for conceivable time (1 category, $numOfLabels labels in the category)', () async {
      final encoder = OneHotEncoder({
        'category': List<String>.generate(numOfLabels, (int idx) => 'label_$idx'),
      });
      final measurer = OneHotEncoderBenchmark(encoder, {'category': ['label_100']});
      final actual = measurer.measure();
      final maxTimeInMicroSeconds = 1500;
      expect(actual, lessThan(maxTimeInMicroSeconds));
    }, skip: true);
  });
}
