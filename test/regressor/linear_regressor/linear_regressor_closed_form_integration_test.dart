import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

void main() {
  group('LinearRegressor with closed form solution', () {
    test('should make correct prediction', () {
      final targetName = 'Grind';
      final data = DataFrame([
        ['Grind', 'RoastLevel', 'Time'],
        [5.5, 5, 20],
        [5.25, 5, 22],
      ]);
      final regressor = LinearRegressor(data, targetName, fitIntercept: false);
      final dataToPredict = [
        [5, 25.0]
      ];
      final dataframeToPredict = DataFrame(dataToPredict, headerExists: false);
      final prediction = regressor.predict(dataframeToPredict);
      final predictedLabel = prediction.rows.first.first;

      expect(predictedLabel, closeTo(5, 0.15));
    });
  });
}
