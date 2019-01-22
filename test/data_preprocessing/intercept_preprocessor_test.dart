import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('Intercept preprocessor', () {
    test('should add intercept to the given points', () {
      final preprocessor = const InterceptPreprocessor(interceptScale: 1.0);
      final processedPoints = preprocessor.addIntercept(
        Float32x4Matrix.from([
          [4.0, 5.0, 10.0],
          [14.0, 49.0, 33.0],
          [41.0, 52.0, 101.0],
        ]),
      );

      expect(
          processedPoints,
          equals([
            [1.0, 4.0, 5.0, 10.0],
            [1.0, 14.0, 49.0, 33.0],
            [1.0, 41.0, 52.0, 101.0]
          ]));
    });

    test('should not mutate given test_data if processing takes place', () {
      final data = Float32x4Matrix.from([
        [4.0, 5.0, 10.0],
        [14.0, 49.0, 33.0],
        [41.0, 52.0, 101.0],
      ]);
      final preprocessor = const InterceptPreprocessor(interceptScale: 1.0);
      final processedPoints = preprocessor.addIntercept(data);

      expect(processedPoints, isNot(same(data)));
    });

    test('should return the same test_data if scale is 0.0 (processing does nnot take place)', () {
      final data = Float32x4Matrix.from([
        [4.0, 5.0, 10.0],
        [14.0, 49.0, 33.0],
        [41.0, 52.0, 101.0],
      ]);
      final preprocessor = const InterceptPreprocessor(interceptScale: 0.0);
      final processedPoints = preprocessor.addIntercept(data);

      expect(processedPoints, same(data));
      expect(processedPoints, [
        [4.0, 5.0, 10.0],
        [14.0, 49.0, 33.0],
        [41.0, 52.0, 101.0]
      ]);
    });

    test('should consider scale parameter (if scale is not equal to 0.0)', () {
      final data = Float32x4Matrix.from([
        [4.0, 5.0, 10.0],
        [14.0, 49.0, 33.0],
        [41.0, 52.0, 101.0],
      ]);
      final preprocessor = const InterceptPreprocessor(interceptScale: -5.0);
      final processedPoints = preprocessor.addIntercept(data);

      expect(processedPoints, [
        [-5.0, 4.0, 5.0, 10.0],
        [-5.0, 14.0, 49.0, 33.0],
        [-5.0, 41.0, 52.0, 101.0]
      ]);
    });
  });
}
